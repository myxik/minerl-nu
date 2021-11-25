def eps_decay(
    steps_done: int, eps_start: float, eps_end: float, step_for_decay: int
) -> float:
    if steps_done > step_for_decay:
        return eps_end
    else:
        eps_decay = (eps_start - eps_end) / step_for_decay
        return eps_start - eps_decay * steps_done
