import sys, yaml, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from nn.architectures import SmallCNN
from data.loaders import load_fashion_mnist
from optim.trainer import evaluate, set_seed
from profiling.reporter import profile_model, save_report_csv

with open('configs/baseline.yaml') as f:
    cfg = yaml.safe_load(f)

ds_cfg = cfg['dataset']
p_cfg = cfg['profiling']
criterion = torch.nn.CrossEntropyLoss()

all_results = []
for seed in [42, 123, 7]:
    set_seed(seed)
    data = load_fashion_mnist(ds_cfg['data_dir'], ds_cfg['batch_size'], ds_cfg['val_fraction'], seed=seed)
    model = SmallCNN(num_classes=cfg['model']['num_classes'], dropout_p=cfg['model']['dropout_p'])
    model_path = f'benchmarks/models/baseline_seed{seed}.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    test_eval = evaluate(model, data['test_loader'], criterion, device="cpu", class_names=data['class_names'])
    prof = profile_model(
        model.eval(),
        tuple(p_cfg['input_shape']),
        p_cfg['batch_size_throughput'],
        p_cfg['latency_warmup'],
        p_cfg['latency_runs'],
        saved_path=model_path,
        label=f'baseline_seed{seed}'
    )
    res = {
        'exp_id': 'E00',
        'method': 'baseline',
        'pruning_ratio': 0.0,
        'quantization': 'fp32',
        'rank_ratio': 1.0,
        'seed': seed,
        'dataset': 'FashionMNIST',
        'epochs': 20,
        'test_accuracy': test_eval['accuracy'],
        'test_macro_f1': test_eval['macro_f1'],
        'test_loss': test_eval['loss'],
        **{k: v for k, v in prof.items() if k != 'label'}
    }
    all_results.append(res)
    print(f"Seed {seed}: Acc={res['test_accuracy']*100:.2f}%, Lat={res['latency_mean_ms']:.3f}ms")

save_report_csv(all_results, 'benchmarks/results/baseline.csv')
print('SUCCESS: all 3 baseline seeds profiled and saved.')
