import os
import numpy as np
import torch
import tqdm

def pth_to_csv(directory, pth_path):
    model = torch.load(pth_path)
    for k, v in tqdm.tqdm(model.items()):
        full_dir = directory
        for l in k.split("."):
            full_dir = os.path.join(full_dir, l)
        vv = v.to('cpu').detach().numpy().copy()
        try:
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
        except OSError:
            print("already exist %s" % full_dir)

        if vv.ndim <= 2:
            np.savetxt(os.path.join(full_dir, "000.csv"), vv, delimiter=",")
        elif vv.ndim == 3:
            for i in range(vv.shape[0]):
                np.savetxt(os.path.join(full_dir, "%03d.csv") % i, vv[i, ...], delimiter=",")
        elif v.ndim == 4:
            for i in range(v.shape[0]):
                for j in range(vv.shape[1]):
                    np.savetxt(os.path.join(full_dir, "%03d_%03d.csv" % (i, j)), vv[i, j, ...], delimiter=",")
        else:
            raise ValueError("Cannot support %d-dimension tensor." % vv.ndim)


def csv_to_pth(directory):
    dic_params = {}
    for curdir, dirs, files in tqdm.tqdm(list(os.walk(directory))):
        csv_files = []
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(file)
        if len(csv_files) > 0:
            key = curdir.lstrip(directory)
            if key[0] == '/':
                key = key[1:]
            name, ext = csv_files[0].split('.')
            fdim = len(name.split('_'))
            if fdim == 1 and len(csv_files) == 1:
                mat = np.loadtxt(os.path.join(curdir, csv_files[0]), delimiter=",", dtype=np.float32)
            elif fdim == 1 and len(csv_files) > 1:
                mat = []
                for f in sorted(csv_files):
                    mat.append(np.loadtxt(os.path.join(curdir, f), delimiter=",", dtype=np.float32))
                mat = np.stack(mat)
            elif fdim == 2:
                mat = []
                rows = 0
                cols = 0
                for f in sorted(csv_files):
                    name, ext = f.split('.')
                    r, c = name.split('_')
                    r, c = int(r), int(c)
                    if r > rows:
                        rows = r
                    if c > cols:
                        cols = c
                    mat.append(np.loadtxt(os.path.join(curdir, f), delimiter=",", dtype=np.float32))
                mat = np.stack(mat)
                if len(mat.shape) == 3:
                    mat = mat.reshape((rows + 1, cols + 1, mat.shape[1], mat.shape[2]))
                elif rows == 0 and cols == 0:# is blank layer
                    mat = mat.reshape((rows + 1, cols + 1, 1, 1))
            dic_params[key.replace("/", ".")] = torch.from_numpy(mat)
    return dic_params


if __name__ == "__main__":
    # check_converter()
    import argparse
    parser = argparse.ArgumentParser(description='csv_serializer')
    subparsers = parser.add_subparsers(dest='command')
    parser_to_csv = subparsers.add_parser('pth_to_csv', help='see `add -h`')
    parser_to_csv.add_argument('input', type=str, help='Path to .pth file')
    parser_to_csv.add_argument('--directory', '-dir', default='test', type=str, help='Path to directory to save csv files')

    parser_to_pth = subparsers.add_parser('csv_to_pth', help='see `commit -h`')
    parser_to_pth.add_argument('output', type=str, help='Path to output .pth file')
    parser_to_pth.add_argument('--directory', '-dir', default='test', type=str, help='Path to directory to load csv files')
    args = parser.parse_args()
    if args.command == 'pth_to_csv':
        pth_to_csv(args.directory, args.input)
    else:
        model = csv_to_pth(args.directory)
        if not os.path.exists(args.output):            
            os.mkdir(args.output)
        pth_name = args.directory.split("/")[-1] + ".pth"
        torch.save(model, os.path.join(args.output, pth_name))