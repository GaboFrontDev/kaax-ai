from app.memory.locks import advisory_lock_key


def test_advisory_lock_key_is_stable_for_same_thread_id() -> None:
    first = advisory_lock_key("thread-123")
    second = advisory_lock_key("thread-123")

    assert first == second


def test_advisory_lock_key_differs_for_different_thread_ids() -> None:
    left = advisory_lock_key("thread-a")
    right = advisory_lock_key("thread-b")

    assert left != right
