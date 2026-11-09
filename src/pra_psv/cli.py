"""Explicit release checks, cached refitting, and DH training."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check release checksums, weights and training partitions")
    commands.add_parser("demo", help="CPU inference on a released training feature pack")
    p = commands.add_parser("train-psv", help="Refit one frozen cached round on CPU")
    p.add_argument("--round", type=int, choices=(0, 1), required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("train-pra-ad", help="Refit PRA on cached AD features")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("train-pra-dh", help="Train PRA on DH with explicit external data")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        from .verification import verify
        print(json.dumps(verify(), indent=2))
    elif args.command == "demo":
        from .assets import pairs
        from .psv.routing import PSVRouter
        row = next(r for r in pairs(1) if r["targets"] is not None)
        decision, prediction = PSVRouter.from_release().select_review(tool="example_action",
            observation_eligible=True, pra_score=row["pack"]["numeric"][0],
            reviews_used=0, pack=row["pack"])
        print(json.dumps(dict(review=decision.review, reasons=decision.reasons,
            prediction=prediction, scope="CPU training-pack demonstration; not benchmark evidence"), indent=2))
    elif args.command == "train-psv":
        from .assets import pairs
        from .psv.model import fit
        import torch
        if args.out.exists():
            parser.error("--out must be a new directory")
        package = fit(pairs(args.round), args.round)
        args.out.mkdir(parents=True, exist_ok=False)
        torch.save(package, args.out / f"router_round{args.round}.pt")
        (args.out / "training_report.json").write_text(json.dumps(package["report"], indent=2) + "\n")
        print(json.dumps(package["report"], indent=2))
    elif args.command == "train-pra-ad":
        from .pra.train_ad import train
        print(json.dumps(train(args.out, device=args.device), indent=2))
    else:
        _train_dh(args)


def _train_dh(args):
    import torch
    from .pra.dh.dataset import load_split
    from .pra.dh.training import load_teacher, prepare, train
    from .pra.model import PrefixFeatureAdapter
    from .pra.encoding import FrozenPrefixEncoder
    if args.out.exists():
        raise FileExistsError("Choose a new output directory")
    split = json.loads(args.split.read_text())
    ids = split["inner_train_ids"]
    records = {r.example_id: r for r in load_split("train", args.data_dir)}
    train_records = [records[i] for i in ids]
    valid_records = [r for r in load_split("val", args.data_dir) if r.in_combined_split]
    if len(train_records) != 2595 or len(valid_records) != 264:
        raise ValueError("Reference DH inventories must contain 2595 train and 264 validation tasks")
    if len(set(ids)) != len(ids) or set(ids) & {r.example_id for r in valid_records}:
        raise ValueError("Invalid DH task partition")
    targets = load_teacher(args.teacher)
    encoder = FrozenPrefixEncoder(args.encoder, max_length=512, device_map=args.device_map)
    training = prepare(train_records, encoder, targets, batch_size=1, training=True)
    validation = prepare(valid_records, encoder, targets, batch_size=1, training=False)
    torch.manual_seed(11)
    model = PrefixFeatureAdapter(encoder.hidden_size, width=512)
    report = train(model, training, validation, epochs=10, batch_size=16, lr=2e-4, seed=11)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(dict(state_dict=model.state_dict(), input_dim=encoder.hidden_size,
                    width=512, encoder=args.encoder), args.out / "adapter.pt")
    (args.out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

