def get_current_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
