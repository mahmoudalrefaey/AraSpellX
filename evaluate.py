from core.args import get_eval_args
from core.evaluate import run_evaluation, print_results


if __name__ == '__main__':
    args = get_eval_args()
    print(args)
    results = run_evaluation(args)
    print_results(results)