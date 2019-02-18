import argparse

def show_args(args):
    if args.opencv:
        print("opencv is used ")
    else:
        print("opencv is not used ")

    print(args.steps)
    print(args.file)
    print(args.data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="learn to use `argparse`")

    # 标志位
    parser.add_argument('--opencv', '-cv', action='store_true', help='use opencv if set ')
    # 必需参数
    parser.add_argument('--steps', '-s', required=True, type=int, help='number of steps')
    # 默认参数
    parser.add_argument('--file', '-f', default='a.txt')
    # 候选参数
    parser.add_argument('--data', '-d', choices=['data1', 'data2'])

    
    args = parser.parse_args()
    show_args(args)

    
