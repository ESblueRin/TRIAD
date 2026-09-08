"""Small argparse CLI. Help and storage inspection do not load a model."""

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="triad",
        description="TRIAD — Every interaction leaves a trace in the weights.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config", type=Path, help="YAML configuration (otherwise built-in defaults)"
    )
    shared.add_argument(
        "--tiny",
        action="store_true",
        help="Random tiny Transformer + scripted Teacher; no downloads",
    )
    shared.add_argument("--data-dir", type=Path, help="Override private runtime storage root")

    chat = commands.add_parser("chat", parents=[shared], help="Interactive online personalization")
    chat.add_argument("--user", required=True)
    chat.add_argument("--debug", action="store_true")
    probe = commands.add_parser(
        "probe", parents=[shared], help="Answer without history or training"
    )
    probe.add_argument("--user", required=True)
    probe.add_argument("prompt")
    info = commands.add_parser(
        "inspect", parents=[shared], help="Inspect resolved LoRA projection paths"
    )
    info.set_defaults(user=None)
    for name, help_text in (
        ("checkpoints", "List retained and pruned adapter versions"),
        ("checkpoint", "Pin the current personalized state"),
        ("rollback", "Restore weights, replay and conversation from a retained version"),
        ("events", "Export local research event records"),
    ):
        command = commands.add_parser(name, parents=[shared], help=help_text)
        command.add_argument("--user", required=True)
        if name == "rollback":
            command.add_argument("--version", type=int, required=True)
        if name == "events":
            command.add_argument("--output", type=Path)
            command.add_argument(
                "--all", action="store_true", help="Include conversations and recovery records"
            )

    evaluation = commands.add_parser(
        "eval", parents=[shared], help="Compare base/current/past without memory"
    )
    evaluation.add_argument("--user", required=True)
    evaluation.add_argument("--dataset", type=Path)
    evaluation.add_argument("--versions", type=int, nargs="*", default=[])
    evaluation.add_argument("--recall", action="store_true")
    evaluation.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    experiment = commands.add_parser(
        "experiment", parents=[shared], help="Run a controlled interaction dataset"
    )
    experiment.add_argument("--user", required=True)
    experiment.add_argument("--dataset", type=Path, required=True)
    experiment.add_argument("--repeats", type=int, default=1)
    experiment.add_argument("--replay", choices=["on", "off"])
    experiment.add_argument("--output", type=Path, default=Path("reports/experiment.json"))
    suite = commands.add_parser(
        "suite", parents=[shared], help="Compare good/noisy/hostile with replay ON/OFF"
    )
    suite.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=[
            Path("examples/good.yaml"),
            Path("examples/noisy.yaml"),
            Path("examples/hostile.yaml"),
        ],
    )
    suite.add_argument("--prefix", required=True, help="Fresh user ID prefix for this study")
    suite.add_argument(
        "--interactions", type=int, default=12, help="Equal interaction budget per condition"
    )
    suite.add_argument("--output", type=Path, default=Path("reports/suite.json"))
    check = commands.add_parser(
        "teacher-check", parents=[shared], help="Validate an actual Teacher response"
    )
    check.add_argument("--query", default="내 이름이 뭐야?")
    check.add_argument("--response", default="성준입니다.")
    check.add_argument("--followup", default="ㅅㅂ 진짜 답답하네.")
    return root


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))


def _debug(event):
    if event is None:
        return
    decision = event.teacher_decision.decision
    print(
        f"[Teacher] type={decision.feedback_type.upper()} confidence={decision.confidence:.3f} "
        f"update_strength={decision.update_strength:.3f} source={event.teacher_decision.source}"
    )
    print(
        f"[TRIAD] training={event.training_performed} loss={event.loss} "
        f"grad_norm={event.gradient_norm} delta={event.training.parameter_delta_norm:.8g} "
        f"lr={event.training.effective_learning_rate:.8g} adapter_version={event.adapter_version}"
    )
    if event.training.error:
        print(f"[TRIAD] {event.training.error}")


def _chat(engine, args):
    print("TRIAD | /exit /new /checkpoint /rollback VERSION /help")
    while True:
        try:
            text = input("You> ")
        except EOFError:
            break
        if text in ("/exit", "/quit"):
            break
        if not text.strip():
            continue
        if text == "/help":
            print(
                "/new: clear chat context, retain weights; /checkpoint: pin state; "
                "/rollback VERSION: restore state; /exit: close (pending turn stays saved)"
            )
            continue
        if text == "/new":
            print(f"[TRIAD] new conversation, version={engine.new_conversation(args.user)}")
            continue
        if text == "/checkpoint":
            version, _ = engine.checkpoint(args.user)
            print(f"[TRIAD] checkpoint={version}")
            continue
        if text.startswith("/rollback "):
            version = engine.rollback(args.user, int(text.split()[1]))
            print(f"[TRIAD] restored as version={version}")
            continue
        result = engine.chat(args.user, text)
        if args.debug:
            _debug(result.event)
        if result.generation_error:
            print(
                f"[TRIAD] Generation failed; any preceding update is saved: {result.generation_error}"
            )
        else:
            print(f"AI> {result.response}")
        if result.event and result.event.training.error and not args.debug:
            print(
                "[TRIAD] Update failed; previous healthy weights retained. See events for details."
            )


def run(args) -> int:
    from .config import load_config
    from .storage import UserStore, write_json

    config = load_config(args.config)
    if args.tiny:
        from .tiny import tiny_config

        if args.config is None:
            config = tiny_config()
        else:
            config.student.model_name = "triad-tiny-random"
            config.student.device = "cpu"
    if args.data_dir:
        config.storage.root = args.data_dir
    if getattr(args, "replay", None) is not None:
        config.replay.enabled = args.replay == "on"
    if args.command in ("checkpoints", "events"):
        store = UserStore(config.storage, args.user)
        with store.lock:
            value = (
                store.checkpoints()
                if args.command == "checkpoints"
                else store.events(
                    None if args.all else "training",
                )
            )
            if getattr(args, "output", None):
                write_json(args.output, value)
                print(args.output)
            else:
                _print(value)
        return 0
    if args.command == "teacher-check":
        from .schemas import Interaction
        from .teacher import OllamaTeacher, TeacherRouter

        if args.tiny:
            from .tiny import DemoTeacher

            teacher = DemoTeacher()
        else:
            teacher = OllamaTeacher(config.teacher)
        result = TeacherRouter(teacher, config.teacher).route(
            Interaction(
                user_query=args.query,
                student_response=args.response,
                user_followup=args.followup,
            )
        )
        _print(result.model_dump())
        return int(result.source == "fallback")

    from .engine import TriadEngine

    if args.tiny:
        import torch

        from .tiny import DemoTeacher, make_tiny_student

        torch.set_num_threads(1)
        engine = TriadEngine(config, make_tiny_student(config), DemoTeacher())
        print(
            "[TRIAD] Tiny random model / scripted Teacher: mechanical demo only.", file=sys.stderr
        )
    else:
        engine = TriadEngine.from_config(config)
    if args.command == "chat":
        _chat(engine, args)
    elif args.command == "probe":
        print(engine.answer(args.user, args.prompt))
    elif args.command == "inspect":
        _print(
            {
                **engine.student.base_metadata,
                "trainable_parameters": sum(
                    p.numel() for p in engine.student.model.parameters() if p.requires_grad
                ),
            }
        )
    elif args.command == "checkpoint":
        version, path = engine.checkpoint(args.user)
        _print({"version": version, "path": path})
    elif args.command == "rollback":
        _print({"restored_as_version": engine.rollback(args.user, args.version)})
    elif args.command == "eval":
        from .evaluation import evaluate
        from .experiments import load_cases

        cases = load_cases(args.dataset) if args.dataset else []
        report = evaluate(engine, args.user, cases, args.versions, include_recall=args.recall)
        write_json(args.output, report)
        _print({name: result["aggregate"] for name, result in report["variants"].items()})
        print(f"Report: {args.output}")
    elif args.command == "experiment":
        from .experiments import load_dataset, run_experiment

        report = run_experiment(
            engine, load_dataset(args.dataset), args.user, args.output, args.repeats
        )
        _print(report["variants"]["current"]["aggregate"])
        print(f"Report: {args.output}")
    elif args.command == "suite":
        from .experiments import load_dataset, run_suite

        report = run_suite(
            engine,
            [load_dataset(path) for path in args.datasets],
            args.prefix,
            args.output,
            args.interactions,
        )
        _print(report["comparison"])
        print(f"Report: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nTRIAD stopped; last committed state is retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"TRIAD error: {exc}", file=sys.stderr)
        return 1
