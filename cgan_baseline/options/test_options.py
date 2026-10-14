import argparse


class TestOptions:
    __test__ = False

    def parse(self, argv=None):
        p = argparse.ArgumentParser(description='Evaluate fixed fold test sets from saved cGAN config/stats')
        p.add_argument('--save_dir', required=True)
        p.add_argument('--checkpoint', choices=['best','final','both'], default='both')
        p.add_argument('--results_dir', default=None)
        p.add_argument('--device', default='cuda')
        p.add_argument('--num_workers', type=int, default=0)
        p.add_argument('--cpu_threads', type=int, default=2)
        p.add_argument('--batch_size', type=int, default=None)
        p.add_argument('--k_folds', type=int, default=None)
        p.add_argument('--excel_path', default=None)
        p.add_argument('--data_root', default=None)
        p.add_argument('--split_file', default=None)
        return p.parse_args(argv)
