"""
Main entry point for the benchmarking tool.

This module provides a command-line interface for running benchmarks using Typer.
It allows users to specify the path to an agent, the benchmark(s) to run, and other
options such as verbosity.

Functions
---------
get_agent : function
    Dynamically imports and returns the default configuration agent from the given path.

main : function
    The main function that runs the specified benchmarks with the given agent.
    Outputs the results to the console.

Attributes
----------
__name__ : str
    The standard boilerplate for invoking the main function when the script is executed.
"""
import importlib
import os.path
import sys

from typing import Annotated, Optional

import typer

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from gpt_engineer.applications.cli.main import load_env_if_needed
from gpt_engineer.benchmark.bench_config import BenchConfig
from gpt_engineer.benchmark.benchmarks.load import get_benchmark
from gpt_engineer.benchmark.run import export_yaml_results, print_results, run

# Import observability
try:
    from gpt_engineer.core.maxim_observability import init_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}
)  # creates a CLI app


def get_agent(path):
    """
    Dynamically imports and returns the default configuration agent from the given path.

    Parameters
    ----------
    path : str
        The file path to the module containing the default configuration agent.

    Returns
    -------
    BaseAgent
        An instance of the imported default configuration agent.
    """
    # Dynamically import the python module at path
    sys.path.append(os.path.dirname(path))
    agent_module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
    return agent_module.default_config_agent()


@app.command(
    help="""
        Run any benchmark(s) against the specified agent.

        \b
        Currently available benchmarks are: apps and mbpp
    """
)
def main(
    path_to_agent: Annotated[
        str,
        typer.Argument(
            help="python file that contains a function called 'default_config_agent'"
        ),
    ],
    bench_config: Annotated[
        str, typer.Argument(help="optional task name in benchmark")
    ] = os.path.join(os.path.dirname(__file__), "default_bench_config.toml"),
    yaml_output: Annotated[
        Optional[str],
        typer.Option(help="print results for each task", show_default=False),
    ] = None,
    verbose: Annotated[
        Optional[bool],
        typer.Option(help="print results for each task", show_default=False),
    ] = False,
    use_cache: Annotated[
        Optional[bool],
        typer.Option(
            help="Speeds up computations and saves tokens when running the same prompt multiple times by caching the LLM response.",
            show_default=False,
        ),
    ] = True,
):
    """
    The main function that runs the specified benchmarks with the given agent and outputs the results to the console.

    Parameters
    ----------
    path_to_agent : str
        The file path to the Python module that contains a function called 'default_config_agent'.
    bench_config : str, default=default_bench_config.toml
        Configuration file for choosing which benchmark problems to run. See default config for more details.
    yaml_output: Optional[str], default=None
        Pass a path to a yaml file to have results written to file.
    verbose : Optional[bool], default=False
        A flag to indicate whether to print results for each task.
    use_cache : Optional[bool], default=True
        Speeds up computations and saves tokens when running the same prompt multiple times by caching the LLM response.
    Returns
    -------
    None
    """
    if use_cache:
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    load_env_if_needed()

    # Initialize observability for benchmark runs
    observability = None
    session_id = None

    if OBSERVABILITY_AVAILABLE:
        try:
            import sys

            from pathlib import Path
            from uuid import uuid4

            observability = init_observability(enabled=True)

            if observability.is_enabled():
                session_id = str(uuid4())

                session_tags = {
                    "agent_path": path_to_agent,
                    "bench_config": str(Path(bench_config).name),
                    "use_cache": str(use_cache),
                    "python_version": sys.version.split()[0],
                    "gpt_engineer_invocation": "benchmark",
                }

                session_metadata = {
                    "benchmark_args": {
                        "path_to_agent": path_to_agent,
                        "bench_config": bench_config,
                        "yaml_output": yaml_output,
                        "verbose": verbose,
                        "use_cache": use_cache,
                    }
                }

                observability.start_session(
                    session_id=session_id, tags=session_tags, metadata=session_metadata
                )

                print(f"Started Maxim benchmark session: {session_id}")

        except Exception as e:
            print(f"Failed to initialize observability: {e}")

    config = BenchConfig.from_toml(bench_config)
    print("using config file: " + bench_config)
    benchmarks = list()
    benchmark_results = dict()
    for specific_config_name in vars(config):
        specific_config = getattr(config, specific_config_name)
        if hasattr(specific_config, "active"):
            if specific_config.active:
                benchmarks.append(specific_config_name)

    try:
        for benchmark_name in benchmarks:
            benchmark = get_benchmark(benchmark_name, config)
            if len(benchmark.tasks) == 0:
                print(
                    benchmark_name
                    + " was skipped, since no tasks are specified. Increase the number of tasks in the config file at: "
                    + bench_config
                )
                continue

            # Start trace for this benchmark
            trace_id = None
            if observability and observability.is_enabled():
                trace_id = str(uuid4())

                trace_tags = {
                    "benchmark_name": benchmark_name,
                    "agent_path": path_to_agent,
                    "task_count": str(len(benchmark.tasks)),
                    "operation": "benchmark_run",
                }

                trace_metadata = {
                    "benchmark_config": getattr(config, benchmark_name).__dict__
                    if hasattr(config, benchmark_name)
                    else {},
                    "task_list": [
                        task.name if hasattr(task, "name") else str(task)
                        for task in benchmark.tasks[:10]
                    ],  # First 10 tasks
                }

                observability.start_trace(
                    trace_id=trace_id,
                    name=f"Benchmark: {benchmark_name}",
                    tags=trace_tags,
                    metadata=trace_metadata,
                    session_id=session_id,
                )

                observability.set_trace_input(
                    f"Running {len(benchmark.tasks)} tasks for benchmark {benchmark_name}"
                )

            agent = get_agent(path_to_agent)

            results = run(agent, benchmark, verbose=verbose)
            print(
                f"\n--- Results for agent {path_to_agent}, benchmark: {benchmark_name} ---"
            )
            print_results(results)
            print()
            benchmark_results[benchmark_name] = {
                "detailed": [result.to_dict() for result in results]
            }

            # End trace for this benchmark
            if observability and observability.is_enabled() and trace_id:
                # Calculate summary statistics
                total_tasks = len(results)
                passed_tasks = sum(
                    1
                    for result in results
                    if hasattr(result, "passed") and result.passed
                )

                output_summary = {
                    "total_tasks": total_tasks,
                    "passed_tasks": passed_tasks,
                    "success_rate": (passed_tasks / total_tasks)
                    if total_tasks > 0
                    else 0,
                    "benchmark_name": benchmark_name,
                }

                observability.set_trace_output(str(output_summary))
                print(f"🔍 Trace Output Set: {output_summary}")
                observability.end_trace(trace_id)

        if yaml_output is not None:
            export_yaml_results(yaml_output, benchmark_results, config.to_dict())

    finally:
        # Cleanup observability resources
        if observability and observability.is_enabled():
            try:
                # End session
                if session_id:
                    observability.end_session()

                # Cleanup SDK
                observability.cleanup()
                print("Maxim benchmark observability cleanup completed")

            except Exception as e:
                print(f"Failed to cleanup benchmark observability: {e}")


if __name__ == "__main__":
    typer.run(main)
