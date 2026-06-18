#!/usr/bin/env python3
"""
自主科研Agent系统 - 主入口

面向金融场景图学习竞赛的自动化实验控制Agent。
支持四阶段科研闭环：文献解析→瓶颈诊断→代码设计→实验验证。

用法:
    # 完整运行（两个任务）
    python main.py --task 1 --task 2 --budget 10

    # 仅运行任务1（分类）
    python main.py --task 1 --budget 5

    # 从配置文件中加载
    python main.py --config config.yaml

    # 恢复之前运行
    python main.py --resume ./output/task1/research_memory.json
"""

import os
import sys
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import SystemConfig, ResearchOrchestrator


def setup_logging(level=logging.INFO):
    """配置日志系统"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="自主科研Agent系统 - 自动化实验控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --task 1 2 --budget 10 --output_dir ./output
  python main.py --config config.yaml
  python main.py --task 2 --budget 5 --device cuda --seed 42
        """
    )

    # 任务选择
    parser.add_argument(
        "--task", type=int, action="append",
        choices=[1, 2],
        help="要运行的任务ID (1=分类, 2=推荐)，可指定多个，如 --task 1 --task 2"
    )

    # Agent配置
    parser.add_argument(
        "--budget", type=int, default=10,
        help="每任务的实验预算轮数 (默认: 10)"
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="总时间限制（秒）(默认: 3600)"
    )
    parser.add_argument(
        "--early_stop", type=int, default=3,
        help="早停耐心值 (默认: 3)"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=20,
        help="最大迭代轮数 (默认: 20)"
    )

    # LLM配置
    parser.add_argument(
        "--llm_model", type=str, default="qwen-turbo",
        help="LLM模型名称 (默认: qwen-turbo)"
    )
    parser.add_argument(
        "--llm_api_key", type=str, default="",
        help="LLM API密钥 (默认从环境变量LLM_API_KEY读取)"
    )
    parser.add_argument(
        "--llm_base_url", type=str, default="",
        help="LLM API基础URL"
    )

    # 系统配置
    parser.add_argument(
        "--config", type=str, default="",
        help="YAML配置文件路径"
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="恢复之前运行的记忆文件路径"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="输出目录 (默认: ./output)"
    )

    # 调试
    parser.add_argument(
        "--debug", action="store_true",
        help="启用调试模式"
    )

    return parser.parse_args()


def build_config(args) -> SystemConfig:
    """根据命令行参数构建系统配置"""
    if args.config and os.path.exists(args.config):
        # 从YAML加载
        config = SystemConfig.from_yaml(args.config)
        print(f"[配置] 已从 {args.config} 加载")
    else:
        # 从参数构建
        config = SystemConfig(
            device=args.device,
            seed=args.seed
        )

    # 覆盖LLM配置
    if args.llm_api_key:
        config.llm.api_key = args.llm_api_key
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    if args.llm_model:
        config.llm.model = args.llm_model

    # 覆盖研究配置
    config.research.budget = args.budget
    config.research.time_limit = args.time_limit
    config.research.early_stop_patience = args.early_stop
    config.research.max_iterations = args.max_iterations
    config.research.output_dir = args.output_dir

    # 确定要运行的任务列表
    task_ids = args.task if args.task else [1, 2]

    # 确保任务配置存在（根据命令行参数填充默认值）
    from agent import TaskConfig
    for task_id in task_ids:
        task_key = f'task{task_id}'
        if task_key not in config.tasks:
            if task_id == 1:
                config.tasks[task_key] = TaskConfig(
                    task_id=1,
                    task_type='classification',
                    data_path='data/cls_data/A1.npz',
                    sample_path='data/cls_data/sample_submission.csv',
                    model_type='sage',
                    output_dir=args.output_dir,
                    epochs=100,
                )
            elif task_id == 2:
                config.tasks[task_key] = TaskConfig(
                    task_id=2,
                    task_type='recommendation',
                    data_path='data/rec_data',
                    sample_path='data/rec_data/sample_submission.csv',
                    model_type='gru4rec',
                    output_dir=args.output_dir,
                    epochs=30,
                )

    # 过滤：只保留命令行指定的任务（当config.yaml包含额外任务时）
    for key in list(config.tasks.keys()):
        if config.tasks[key].task_id not in task_ids:
            del config.tasks[key]

    # 统一所有任务的 output_dir
    for task_cfg in config.tasks.values():
        task_cfg.output_dir = args.output_dir

    return config


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("  自主科研Agent系统 - 自动化实验控制")
    print("=" * 60)

    # 构建配置
    config = build_config(args)

    # 输出配置摘要
    task_ids = args.task if args.task else [1, 2]
    print(f"\n[配置摘要]")
    print(f"  任务: {task_ids}")
    print(f"  预算: {config.research.budget} 轮/任务")
    print(f"  时间限制: {config.research.time_limit} 秒")
    print(f"  设备: {config.device}")
    print(f"  随机种子: {config.seed}")
    print(f"  输出目录: {config.research.output_dir}")
    print(f"  LLM模型: {config.llm.model}")
    print(f"  调试模式: {'开启' if args.debug else '关闭'}")

    # 创建编排器
    orchestrator = ResearchOrchestrator(config, task_ids=task_ids)

    # 如果指定了恢复文件
    if args.resume:
        if os.path.exists(args.resume):
            orchestrator.memory.load(path=args.resume)
            print(f"\n[恢复] 已从 {args.resume} 加载记忆")
        else:
            print(f"\n[警告] 恢复文件不存在: {args.resume}")

    # 运行科研流程
    print(f"\n{'=' * 60}")
    print("  开始科研流程")
    print(f"{'=' * 60}\n")

    try:
        results = orchestrator.run()

        print(f"\n{'=' * 60}")
        print("  科研流程完成")
        print(f"{'=' * 60}")
        print(f"\n[结果摘要]")
        for task_id, result in results.get("task_results", {}).items():
            print(f"  任务 {task_id}: {result}")
        if "submission_dir" in results:
            print(f"\n  提交目录: {results['submission_dir']}")
        print(f"  总耗时: {results.get('total_time', 0):.1f} 秒")

    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动停止")
        # 保存当前进度
        for task_id in task_ids:
            orchestrator.memory.save(task_id)
        print(f"[保存] 进度已保存到 {config.research.output_dir}")
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=args.debug)
        raise


if __name__ == "__main__":
    main()
