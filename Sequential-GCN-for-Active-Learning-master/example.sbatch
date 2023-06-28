#!/bin/bash
#SBATCH --job-name=example_job     # 作业名称
#SBATCH --nodes=1                  # 单节点
#SBATCH --ntasks-per-node=1        # 单节点运行一个任务
#SBATCH --time=01:00:00            # 运行时间限制为 1 小时
#SBATCH --output=example.out       # 输出文件名
#SBATCH --error=example.err        # 错误输出文件名

echo "Hello, world!"
