Environments
============

Built-in environments
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 22 48

   * - Name
     - Agents
     - Config
     - Description
   * - ``language``
     - 2+
     - ``LanguageTaskConfig``
     - Sequential collaboration on QA (HotPotQA), math, creative writing, and coding
       (HumanEval), selected with ``task_type``.
   * - ``overcooked``
     - 2
     - ``OvercookedConfig``
     - Cooperative cooking with sparse team rewards and role differentiation;
       ``layout`` selects the kitchen.
   * - ``pistonball``
     - 10--20
     - ``PistonballConfig``
     - Large-scale cooperative control under partial observability;
       ``num_pistons`` sets the team size.

.. code-block:: python

   env = langmarl.make_env("language", config)
   langmarl.list_envs()   # every registered name, including your own

Registering your own
--------------------

Any environment is a plugin: subclass ``BaseEnvironment``, decorate it with
``@register_env``, and it becomes available to ``make_env`` and to JSON configs.

The one method that matters is ``collect_trajectory`` -- it runs a full episode
given the current language policies and returns a ``Trajectory`` carrying the
episode reward. Everything else in LangMARL works off that trajectory.

.. code-block:: python

   import langmarl

   @langmarl.register_env("my_env")
   class MyEnv(langmarl.BaseEnvironment):
       def __init__(self, config):
           self.num_agents = config.num_agents
           self.llm_client = langmarl.LLMClient(config.get_actor_llm())

       def sample_tasks(self, num_samples: int) -> list[dict]:
           return [{"question": "What is 2+2?", "ground_truth": "4"}] * num_samples

       def collect_trajectory(self, policies: dict[str, str], task: dict):
           steps = []
           for agent, policy in policies.items():
               response = self.llm_client.chat(policy, task["question"])
               steps.append({"agent_id": agent, "input": task["question"], "output": response})
           reward = 1.0 if task["ground_truth"] in steps[-1]["output"] else 0.0
           return langmarl.Trajectory(task=task, steps=steps, reward=reward)

       def reset(self, task: dict) -> dict:
           return {"task": task}

       def step(self, agent_id: str, action: str):
           return {}, 0.0, False, {}

Train on it with the standard components:

.. code-block:: python

   config = langmarl.BaseConfig(
       paradigm="central_credit",
       llm=langmarl.LLMConfig.from_preset("gpt-4o-mini"),
   )
   env = langmarl.make_env("my_env", config)

.. tip::

   Rewards can be sparse and binary. The critic converts a single episode-level
   reward into per-agent language feedback, which is what actually drives learning.
