#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import statistics

import ray
from ray.exceptions import RayError

try:
    from ..utils.logger import print_log
except ImportError:
    try:
        from utils.logger import print_log
    except ImportError:
        def print_log(msg):
            print(msg)

try:
    from .base import TaskScanner, PepTaskScanner, run_pyrosetta
except ImportError:
    from base import TaskScanner, PepTaskScanner, run_pyrosetta

# @ray.remote(num_gpus=1/8, num_cpus=1)
# def run_openmm_remote(task):
#     return run_openmm(task)


@ray.remote(num_cpus=1)
def run_pyrosetta_remote(task):
    return run_pyrosetta(task)


def parse():
    parser = argparse.ArgumentParser(description='calculating dG using pyrosetta')
    parser.add_argument('--results', type=str, required=True, help='Path to the summary of the results (.jsonl)')
    parser.add_argument('--n_sample', type=int, default=float('inf'), help='Maximum number of samples for calculation')
    parser.add_argument('--rfdiff_relax', action='store_true', help='Use rfdiff fastrelax')
    parser.add_argument('--out_path', type=str, default=None, help='Output path, default dG_report.jsonl under the same directory as results')
    parser.add_argument('--failed_out_path', type=str, default=None, help='Output path for failed samples (.jsonl)')
    parser.add_argument('--num_workers', type=int, default=2, help='Max concurrent Ray workers for PyRosetta tasks')
    parser.add_argument('--max_retries', type=int, default=1, help='Retries for a crashed Ray task')
    return parser.parse_args()


def main(args):
    # output summary
    if args.out_path is None:
        # args.out_path = os.path.join(os.path.dirname(args.results), 'dG_report.jsonl')
        _dir = os.path.dirname(args.results)
        _base = os.path.basename(args.results)
        db, _, src = os.path.splitext(_base)[0].split('_')
        args.out_path = os.path.join(_dir, f'{db}_rosetta_dG_report_{src}.jsonl')
        print("output path: ",args.out_path)

    if args.failed_out_path is None:
        _dir = os.path.dirname(args.out_path)
        _base = os.path.basename(args.out_path)
        name, _ = os.path.splitext(_base)
        args.failed_out_path = os.path.join(_dir, f'{name}_failed.jsonl')
    print("failed path: ", args.failed_out_path)
        
    results = {}
    failed_records = []

    # parallel
    ray.init(num_cpus=args.num_workers)
    # scanner = TaskScanner(args.results, args.n_sample, args.rfdiff_relax)
    scanner = PepTaskScanner(args.results, args.n_sample, args.rfdiff_relax)
    if args.results.endswith('txt'):
        tasks = scanner.scan_dataset()
    else:
        tasks = scanner.scan()
    future_to_task = {}
    for task in tasks:
        fut = run_pyrosetta_remote.options(max_retries=args.max_retries).remote(task)
        future_to_task[fut] = task
    futures = list(future_to_task.keys())
    open(args.failed_out_path, 'w').close()
    if len(futures) > 0:
        print_log(f'Submitted {len(futures)} tasks.')
    failed = 0
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            src_task = future_to_task.pop(done_id, None)
            try:
                done_task = ray.get(done_id)
            except RayError as e:
                failed += 1
                failed_record = {
                    'type': 'ray_worker_crash',
                    'in_path': getattr(src_task, 'in_path', None),
                    'current_path': getattr(src_task, 'current_path', None),
                    'id': (src_task.info.get('id') if src_task is not None else None),
                    'number': (src_task.info.get('number') if src_task is not None else None),
                    'error': str(e),
                }
                failed_records.append(failed_record)
                with open(args.failed_out_path, 'a') as f:
                    f.write(json.dumps(failed_record) + '\n')
                print_log(f'Remaining {len(futures)}. Failed one task due to Ray worker error: {e}')
                continue
            print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}, dG {done_task.dG}')
            if done_task.status == 'failed':
                failed_record = {
                    'type': 'pyrosetta_runtime_failure',
                    'in_path': done_task.in_path,
                    'current_path': done_task.current_path,
                    'id': done_task.info.get('id'),
                    'number': done_task.info.get('number'),
                    'error': 'pyrosetta task returned failed status',
                }
                failed_records.append(failed_record)
                with open(args.failed_out_path, 'a') as f:
                    f.write(json.dumps(failed_record) + '\n')
            _id, number = done_task.info['id'], done_task.info['number']
            if _id not in results:
                results[_id] = {
                    'min': float('inf'),
                    'all': {}
                }
            results[_id]['all'][number] = done_task.dG
            results[_id]['min'] = min(results[_id]['min'], done_task.dG)
    
    # write results
    for _id in results:
        success = 0
        for n in results[_id]['all']:
            if results[_id]['all'][n] < 0:
                success += 1
        results[_id]['success rate'] = success / len(results[_id]['all'])
    json.dump(results, open(args.out_path, 'w'), indent=2)
    if failed > 0:
        print_log(f'Warning: {failed} tasks failed permanently and were skipped.')
    print_log(f'Failed sample records: {len(failed_records)} written to {args.failed_out_path}')

    # show results
    vals = [results[_id]['min'] for _id in results]
    if len(vals) == 0:
        print('No successful task to summarize.')
        return
    print(f'median: {statistics.median(vals)}, mean: {sum(vals) / len(vals)}')
    success = [results[_id]['success rate'] for _id in results]
    print(f'mean success rate: {sum(success) / len(success)}')


if __name__ == '__main__':
    import random
    random.seed(12)
    main(parse())
