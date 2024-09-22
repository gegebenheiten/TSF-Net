import argparse

class SimpleOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simple Options for Training')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--initial_lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--num_epochs', type=int, default=36, help='number of epochs to train')
        self.parser.add_argument('--senarios', type=str, nargs='+', default=['ball_00', 'ball_02', 'basket_05', 'eggs_01', 'eggs_03', 'eggs_05', 'horse_03', 'may29_handheld_02', 'may29_handheld_04'], help='list of scenarios')
        self.parser.add_argument('--data_root_dir', type=str, default='/home/nwn9209/tsfnet/tsfnet_code/EXP1_dataset', help='root directory for data')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        return self.opt