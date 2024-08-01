import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='Physics-infromed AE for battery SoH !')
    parser.add_argument('--input_channel',default=3)
    # method type
    parser.add_argument('--method_type', type=str, default='lstm')

    # data
    parser.add_argument('--source_dir',type=str, default='data/CALCE_Anysis/AE_input/CS2')
    parser.add_argument('--test_id',type=int, default=2,choices=[1,2,3,4])
    parser.add_argument('--normalize_type',default='minmax',choices=['minmax','standard'])
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--seed',default=2022)
    parser.add_argument('--num_samples', default=10)

    # optimizer
    parser.add_argument('--lr',type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler
    parser.add_argument('--lr_step', type=list,default=[1000,2000])
    parser.add_argument('--lr_gamma',type=float,default=0.5)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--n_epoch',default=20000)
    parser.add_argument('--early_stop', default=2000)
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--beta',default=0.9)

    # reulsts
    parser.add_argument('--is_plot_test_results',default=False)
    parser.add_argument('--is_save_logging',default=True)
    parser.add_argument('--is_save_best_model',default=True)
    parser.add_argument('--is_save_to_txt',default=True)
    parser.add_argument('--is_save_test_results',default=True)
    parser.add_argument('--experiment_time',type=int,default=1)
    parser.add_argument('--isTest',type=bool,default=False)
    parser.add_argument('--save_root',default='experiment/CALCE')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for i in args.items():
        print(i)
