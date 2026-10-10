"""Evidence-grounded CRM decision agent and its baselines.

- ``src.agent.policy``     -- versioned constants, permission classes and the safety guard
- ``src.agent.tools``      -- local tools (read / compute / local write); external tools are never executed
- ``src.agent.planner``    -- information-need tool selection and evidence-gated final actions
- ``src.agent.controller`` -- the bounded observe-plan-act-evaluate loop, traces and batch prioritization
- ``src.agent.assistant``  -- Baseline B: the PR #4 rule-based assistant
- ``src.agent.experiment`` -- baseline comparison, ablations and outcome-masked replay
"""
