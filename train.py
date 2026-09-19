from core.args import get_train_args
from core.train import main

if __name__ == '__main__':
    args = get_train_args()
    print(args)
    main(args)
