"""Compare frozen baseline/candidate; calibrate suggestions on validation only."""

import argparse
import json

import numpy as np
import torch

from .config import ARTIFACTS, DATA
from .photo_tags import LABELS, PhotoTagCNN
from .photo_train import predict, scores
from .train import save_checkpoint, save_json


def suggestion_thresholds(probabilities, target):
    thresholds, evidence = [], []
    for i in range(len(LABELS)):
        options = []
        positives = int(target[:, i].sum())
        if positives >= 5:
            for threshold in np.arange(0.1, 0.951, 0.025):
                selected = probabilities[:, i] >= threshold
                tp = int((selected & (target[:, i] > 0.5)).sum())
                precision = tp / max(int(selected.sum()), 1)
                if tp >= 5 and precision >= 0.7:
                    options.append((tp / positives, precision, float(threshold), tp, int(selected.sum())))
        if options:
            recall, precision, threshold, tp, count = max(options)
            thresholds.append(threshold)
            evidence.append({"validation_precision": precision, "validation_recall": recall, "tp": tp, "predictions": count})
        else:
            thresholds.append(1.1)  # No evidence for automatic suggestions; manual selection remains available.
            evidence.append(None)
    return thresholds, evidence


def average_precision(probabilities, target):
    values = []
    for i in range(target.shape[1]):
        count = int(target[:, i].sum())
        if not count:
            continue
        order = np.argsort(-probabilities[:, i], kind='stable')
        sorted_scores = probabilities[order, i]
        cumulative = (target[order, i] > 0.5).cumsum()
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1]
        recall = cumulative[ends] / count
        precision = cumulative[ends] / (ends + 1)
        values.append(float((np.diff(np.r_[0, recall]) * precision).sum()))
    return {"mean_average_precision": float(np.mean(values)), "supported_classes": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    images = {s: np.load(DATA / f'coco/{s}-64.npy', mmap_mode='r') for s in ('val', 'test')}
    target = {s: np.load(DATA / f'photo_tags/{s}-labels.npy') for s in images}
    report = {}
    candidate_record = None
    for name, folder in (('baseline', 'photo_tags'), ('candidate', 'photo_tags_improved')):
        record = torch.load(ARTIFACTS / folder / 'best.pt', map_location='cpu', weights_only=True)
        model = PhotoTagCNN(**record['config']).eval()
        model.load_state_dict(record['model'])
        val_probs = predict(model, images['val'])
        thresholds, evidence = suggestion_thresholds(val_probs, target['val'])
        test_probs = predict(model, images['test'])
        # Production additionally suppresses low-support classes, then limits to top five scores.
        production = np.asarray(thresholds)
        for i, label in enumerate(LABELS):
            if record['manifest']['splits']['train']['positive_counts'][label] < 100:
                production[i] = 1.1
        masked = test_probs.copy()
        for row in masked:
            eligible = np.flatnonzero(row >= production)
            selected = eligible[np.argsort(-row[eligible])[:5]]
            row[np.setdiff1d(np.arange(len(row)), selected)] = -1
        selected = masked >= production
        tp = int((selected & (target['test'] > 0.5)).sum())
        count = int(selected.sum())
        report[name] = {'step': record['step'], 'validation': scores(val_probs, target['val'], record['thresholds']),
                        'test': scores(test_probs, target['test'], record['thresholds']),
                        'ranking': average_precision(test_probs, target['test']),
                        'suggestions': {'precision': tp / max(count, 1), 'recall': tp / max(int(target['test'].sum()), 1),
                                        'predictions': count, 'images_with_suggestions': int(selected.any(1).sum())}}
        report[name]['test'].update({'split': 'test', 'step': record['step'], 'images': len(images['test']),
                                    'threshold_source': 'validation only'})
        if name == 'baseline':
            previous_thresholds = np.maximum(0.6, record['thresholds'])
            for i, label in enumerate(LABELS):
                if (record['manifest']['splits']['train']['positive_counts'][label] < 100
                        or record['manifest']['splits']['val']['positive_counts'][label] < 5):
                    previous_thresholds[i] = 1.1
            old_selected = test_probs >= previous_thresholds
            for i in range(len(old_selected)):
                eligible = np.flatnonzero(old_selected[i])
                keep = eligible[np.argsort(-test_probs[i, eligible])[:5]]
                old_selected[i] = False
                old_selected[i, keep] = True
            old_tp = int((old_selected & (target['test'] > 0.5)).sum())
            report[name]['original_runtime_suggestions'] = {
                'precision': old_tp / max(int(old_selected.sum()), 1),
                'recall': old_tp / max(int(target['test'].sum()), 1),
                'predictions': int(old_selected.sum())}
        if name == 'candidate':
            record['suggestion_thresholds'] = thresholds
            record['suggestion_evidence'] = evidence
            candidate_record = record
    improved = report['candidate']['validation']['macro_f1'] > report['baseline']['validation']['macro_f1']
    report['promotion_rule'] = 'Validation macro F1 only; test results do not select checkpoints or thresholds.'
    report['promoted'] = bool(args.promote and improved)
    if args.promote and improved:
        backup = ARTIFACTS / 'photo_tags_baseline'
        backup.mkdir(exist_ok=True)
        for name in ('best.pt', 'last.pt', 'history.json', 'test_metrics.json'):
            source = ARTIFACTS / 'photo_tags' / name
            destination = backup / name
            if source.exists() and not destination.exists():
                destination.write_bytes(source.read_bytes())
        save_checkpoint(ARTIFACTS / 'photo_tags/best.pt', candidate_record)
        for name in ('last.pt', 'history.json'):
            source = ARTIFACTS / 'photo_tags_improved' / name
            if source.exists():
                temporary = (ARTIFACTS / 'photo_tags' / name).with_suffix('.promoting')
                temporary.write_bytes(source.read_bytes())
                temporary.replace(ARTIFACTS / 'photo_tags' / name)
        save_json(ARTIFACTS / 'photo_tags/test_metrics.json', report['candidate']['test'])
    save_json(ARTIFACTS / 'photo_tags_improved/comparison.json', report)
    print(json.dumps({name: {'step': value['step'], 'val_macro': value['validation']['macro_f1'],
                            'test_macro': value['test']['macro_f1'], 'test_micro': value['test']['micro_f1'],
                            'ranking':value['ranking'], 'suggestions':value['suggestions']} for name,value in report.items() if isinstance(value,dict)} | {'promoted':report['promoted']}), flush=True)


if __name__ == '__main__':
    main()
