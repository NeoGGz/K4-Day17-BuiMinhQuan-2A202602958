from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with low compaction threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parent.parent / "data"

    provider_cfg = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=50,  # Small threshold to quickly trigger compaction
        compact_keep_messages=2,
        model=provider_cfg,
        judge_model=provider_cfg,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)

    user_id = "test_user"
    initial_content = "# User Profile: test_user\n\n- **Nơi ở**: Đà Nẵng\n"
    path = store.write_text(user_id, initial_content)

    assert path.exists()
    assert store.path_for(user_id) == path

    # Test read
    content = store.read_text(user_id)
    assert "Đà Nẵng" in content

    # Test edit
    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    updated_content = store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content

    # Test upsert_fact and facts()
    store.upsert_fact(user_id, "Nghề nghiệp", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("Nơi ở") == "Huế"
    assert facts.get("Nghề nghiệp") == "MLOps engineer"
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    thread_id = "thread_compact_test"
    user_id = "user_compact"

    for i in range(8):
        agent.reply(
            user_id=user_id,
            thread_id=thread_id,
            message=(
                f"Lượt {i}: Đây là một tin nhắn tương đối dài để đảm bảo lượng token vượt ngưỡng compact 50 token. "
                "Hệ thống cần kích hoạt compact memory để tóm tắt lịch sử cũ."
            ),
        )

    # Check that compactions were triggered
    assert agent.compaction_count(thread_id) > 0
    ctx = agent.compact_memory.context(thread_id)
    assert len(str(ctx.get("summary", ""))) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions/threads while baseline forgets."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct"

    # Thread 1: Introduction of facts
    intro_msg = "Chào bạn, mình tên là DũngCT, hiện đang ở Huế và làm MLOps engineer."
    baseline.reply(user_id=user_id, thread_id="thread-session-1", message=intro_msg)
    advanced.reply(user_id=user_id, thread_id="thread-session-1", message=intro_msg)

    # Thread 2: Fresh thread testing recall
    query_msg = "Hiện tại mình làm nghề gì và đang ở đâu?"
    base_res = baseline.reply(user_id=user_id, thread_id="thread-session-2", message=query_msg)
    adv_res = advanced.reply(user_id=user_id, thread_id="thread-session-2", message=query_msg)

    base_ans = base_res["response"]
    adv_ans = adv_res["response"]

    # Baseline has NO memory across sessions
    assert "MLOps engineer" not in base_ans
    assert "Huế" not in base_ans

    # Advanced recalls persistent facts from User.md across sessions
    assert "MLOps engineer" in adv_ans
    assert "Huế" in adv_ans


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    thread_id = "thread_stress_load"
    user_id = "dungct_stress"

    messages = [
        f"Turn {i}: Đoạn văn bản dài để kiểm tra mức tiêu thụ prompt tokens qua từng lượt hội thoại. "
        "Chi tiết kỹ thuật liên quan đến Artemis III, X-59 siêu thanh, WMO biến đổi khí hậu và năng lượng British Columbia."
        for i in range(12)
    ]

    for msg in messages:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced agent should process significantly fewer cumulative prompt tokens due to compaction
    assert adv_prompt_tokens < base_prompt_tokens
    assert advanced.compaction_count(thread_id) > 0


def test_confidence_threshold_and_conflict_handling(tmp_path: Path) -> None:
    """Verify confidence gating and conflict resolution for updates."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_bonus"

    # High confidence fact should be written
    store.upsert_fact(user_id, "Nơi ở", "Đà Nẵng", confidence=0.95, min_confidence=0.6)
    facts = store.facts(user_id)
    assert facts.get("Nơi ở") == "Đà Nẵng"

    # Low confidence fact should be rejected
    store.upsert_fact(user_id, "Nơi ở", "Hà Nội", confidence=0.4, min_confidence=0.6)
    facts = store.facts(user_id)
    assert facts.get("Nơi ở") == "Đà Nẵng"  # Not overwritten by low-confidence fact

    # Conflict handling: new authoritative correction with high confidence updates the fact
    store.upsert_fact(user_id, "Nơi ở", "Huế", confidence=1.0, min_confidence=0.6)
    facts = store.facts(user_id)
    assert facts.get("Nơi ở") == "Huế"


def test_noise_filtering_and_question_skipping(tmp_path: Path) -> None:
    """Verify noise filter ignores jokes/temporary trips and skips pure questions."""
    from memory_store import extract_profile_updates

    # Pure questions should produce empty updates
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Mình tên gì và đồ uống yêu thích là gì?") == {}
    assert extract_profile_updates("Hiện tại mình đang ở đâu?") == {}

    # Noise rejection: jokes should not be accepted as profession
    joke_msg = (
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager "
        "cho đỡ phải ngồi canh pipeline, nhưng đó chỉ là câu đùa. Nghề nghiệp hiện tại vẫn là MLOps engineer."
    )
    updates = extract_profile_updates(joke_msg)
    assert updates.get("Nghề nghiệp") == "MLOps engineer"
    assert "product manager" not in str(updates.values())

    # Temporary trip should not be accepted as current location
    trip_msg = (
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại. "
        "Mình đang làm việc ở Đà Nẵng vài tháng."
    )
    updates_trip = extract_profile_updates(trip_msg)
    assert updates_trip.get("Nơi ở") == "Đà Nẵng"
    assert "Hà Nội" not in str(updates_trip.values())


def test_memory_decay(tmp_path: Path) -> None:
    """Verify memory decay prunes obsolete or unrefreshed facts."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_decay"

    store.upsert_fact(user_id, "Tên", "DũngCT")
    store.upsert_fact(user_id, "Nghề nghiệp", "MLOps engineer")
    store.upsert_fact(user_id, "Sở thích cũ", "Chơi game bài")

    assert len(store.facts(user_id)) == 3

    # Retain active facts and decay unconfirmed facts
    store.apply_decay(user_id, retained_keys=["Tên", "Nghề nghiệp"])
    decayed_facts = store.facts(user_id)

    assert "Tên" in decayed_facts
    assert "Nghề nghiệp" in decayed_facts
    assert "Sở thích cũ" not in decayed_facts
