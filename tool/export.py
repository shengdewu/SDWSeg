import argparse
import warnings

import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def export_formats():
    x = [
        ['PyTorch', 'pt', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
    ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def export_pt(model, file):
    f = file.replace('.pth', '.pt')
    torch.save(model, f)
    return f, None


def export_torchscript(model, im, file, optimize):
    f = file.replace('.pth', '.torchscript')

    ts = torch.jit.trace(model, im, strict=False)
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimize_for_mobile(ts)._save_for_lite_interpreter(f)
    else:
        ts.save(f)
    return f, None


def export_onnx(model, im, file, opset, dynamic, simplify):
    import onnx
    f = file.replace('.pth', '.onnx')

    output_names = ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if simplify:
        try:
            # 'onnx-simplifier>=0.4.1')
            import onnxsim

            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'simplifier failure: {e}')
    return f, model_onnx


def main(config: str,
         weight: str,
         imgsz=608,  # image (height, width)
         device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
         optimize=False,
         simplify=False,  # ONNX: simplify model
         opset=12,  # ONNX: opset version
         include=('torchscript', 'onnx'),  # include formats
         ):
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    pt, jit, onnx = flags  # export booleans

    from tool.create_encoder_decoder import create_encoder_decoder
    from tool.performance import test_performance

    model = create_encoder_decoder(config, weight, device)
    test_performance(model, imgsz)

    # Exports
    im = torch.zeros(1, 3, imgsz, imgsz).to(device)
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if pt:
        f[0], _ = export_pt(model, weight)
    if jit:  # TorchScript
        f[1], _ = export_torchscript(model, im, weight, optimize)
    if onnx:  # ONNX
        f[2], _ = export_onnx(model, im, weight, opset, False, simplify)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='config')
    parser.add_argument('--weight', type=str, help='model.pt path(s)')
    parser.add_argument('--imgsz', type=int, default=608, help='image (h, w)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['torchscript'],
        help='torchscript, onnx')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))
