# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.env import Envs


# Every Relax driver/worker imports this package before creating TQ clients.
# Reference bodies follow TQ cleanup; full tensor delivery needs no hook.
if Envs.RELAX_REPLAY_REFERENCES:
    from relax.distributed.replay_manager.core import install_tq_cleanup_hook

    install_tq_cleanup_hook()
