from rl_mdp.util import create_policy_1, create_policy_2, create_mdp
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator


def main():
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    policy1 = create_policy_1()
    policy2 = create_policy_2()
    env = create_mdp()
    mc_evaluator = MCEvaluator(env)
    td_evaluator = TDEvaluator(env, alpha=0.1)
    td_lambda_evaluator = TDLambdaEvaluator(env, alpha=0.1, lambd=0.5)
    num_episodes = 1000

    V_pi1_mc = mc_evaluator.evaluate(policy1, num_episodes)
    print("MC V_pi1:", V_pi1_mc)
    V_pi2_mc = mc_evaluator.evaluate(policy2, num_episodes)
    print("MC V_pi2:", V_pi2_mc)

    V_pi1_td = td_evaluator.evaluate(policy1, num_episodes)
    print("TD V_pi1:", V_pi1_td)
    V_pi2_td = td_evaluator.evaluate(policy2, num_episodes)
    print("TD V_pi2:", V_pi2_td)

    V_pi1_td_lambda = td_lambda_evaluator.evaluate(policy1, num_episodes)
    print("TD lambda V_pi1:", V_pi1_td_lambda)
    V_pi2_td_lambda = td_lambda_evaluator.evaluate(policy2, num_episodes)
    print("TD lambda V_pi2:", V_pi2_td_lambda)


if __name__ == "__main__":
    main()
