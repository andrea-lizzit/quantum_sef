#!/usr/bin/env python

import numpy as np
from neural.models import ConvSEContX3L
from qe_utils import QEDir, load_qe_se
import fit
import logging
from pade.model_pade import AverageModel, AverageSimilarModel, AverageLSModel
from neural.storagemanager import StorageManager

RY = 13.605693122994


def dyson(E0, correlation, precision=0.000001):
    E = E0
    for _ in range(10):
        print(f"new E: {E}")
        prev_E = E
        E = E0 + correlation(E)
        print(f"self energy {correlation(E)}")
        if abs(prev_E - E) < precision * abs(E0):
            pass
        if abs(prev_E - E) > 10 * abs(E0):
            raise RuntimeError("self-consistent calculation diverges")
    return E


def dyson_s(E0, e0, correlation, precision=0.000001):
    """Solve the quasiparticle equation
    Args:
            E0(float): eigenvalue of the unperturbed Hamiltonian
            e0(float): starting value for the iterative solution
            correlation: self-energy (correlation part?)
            precision(float): target precision for the solution"""
    E = e0
    for _ in range(10):
        # 		print(f"new E: {E}")
        prev_E = E
        E = E0 + correlation(E)
        if abs(prev_E - E) < precision * abs(E0):
            pass
        if abs(E) > 10 * abs(E0):
            raise RuntimeError("self-consistent calculation diverges")
    return E


def GW(qe_E, correlation, offset):
    E0 = qe_E["DFT"] / RY - offset
    E0 = complex(E0, 0)
    E0 = qe_E["HF-pert"] / RY - offset + correlation(E0)

    E = dyson_s(qe_E["HF-pert"] / RY - offset, E0, correlation)
    E += offset
    E *= RY
    return E


def GW_classic(qe_E, correlation, offset):
    E = dyson_s(
        qe_E["HF-pert"] / RY - offset, qe_E["HF-pert"] / RY - offset, correlation
    )
    E += offset
    E *= RY
    return E


def plot_se(qedir, se, orbital):
    # 	prefix, suffix = args.self_energy.split(",")
    # filename_real, filename_imag = prefix + "-re_on_im" + suffix, prefix + "-im_on_im" + suffix
    # orbital_i = int(suffix)
    # qe_data = load_qe_se(filename_real, filename_imag, positive=True)
    # z, s = qe_data["z"], qe_data["s"]
    # fit_s = [get_model(orbital_i)(vz) for vz in z]
    qe_data = qedir.get_se(orbital, positive=True)
    z, s = qe_data["z"], qe_data["s"]
    fit_s = [se(vz) for vz in z] # fit on the imaginary axis
    z, s, fit_s = np.array(z), np.array(s), np.array(fit_s)
    fig, ax = plt.subplots(2)
    # imaginary axis
    ax[0].plot(np.imag(z), np.real(s), label="reference s real")
    ax[0].plot(np.imag(z), np.imag(s), label="reference s imag")
    ax[0].plot(np.imag(z), np.real(fit_s), dashes=[2, 2], label="fit real")
    ax[0].plot(np.imag(z), np.imag(fit_s), dashes=[2, 2], label="fit imag")
    ax[0].plot(np.imag(z), np.real(qe_data["fit_imag"]), dashes=[3, 2], label="fit (qe) re")
    ax[0].plot(np.imag(z), np.imag(qe_data["fit_imag"]), dashes=[3, 2], label="fit (qe) im")
    ax[1].set_xlabel("x (Ry)")
    ax[1].set_ylabel("Σ(ix) (Ry)")
    ax[0].legend()
    # real axis
    ax[1].plot(np.imag(z), np.real(qe_data["fit_real"]), label="fit (qe) re")
    ax[1].plot(np.imag(z), np.imag(qe_data["fit_real"]), label="fit (qe) im")
    y = list(map(se, z * (-1j)))
    ax[1].plot(np.imag(z), np.real(y), dashes=[2, 2], label="fit re")
    ax[1].plot(np.imag(z), np.imag(y), dashes=[2, 2], label="fit im")
    ax[1].set_xlabel("x (Ry)")
    ax[1].set_ylabel("Σ(x) (Ry)")
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    import csv, argparse, yaml
    import matplotlib.pyplot as plt
    from jax.config import config

    config.update("jax_enable_x64", True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("quantum-sef.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    parser.add_argument("--orbital", type=int)
    parser.add_argument("--plot", action="store_true")
    subparsers = parser.add_subparsers(dest="subparser")
    parser.add_argument("dir", help="directory containing the quantum espresso files")
    p_loadparams = subparsers.add_parser("gwwparams")
    p_loadparams.add_argument(
        "--self_energy", "-s", help="qe self-energy file for comparison"
    )
    p_multipole = subparsers.add_parser("multipole")
    p_multipole.add_argument("n_poles", type=int)
    p_pade = subparsers.add_parser("pade")
    p_pade.add_argument("--type", choices=["avgLS", "avgsimilar"])
    p_pade.add_argument("--precise_iter", type=int, default=0)
    p_a = subparsers.add_parser("gwwfit")
    p_a.add_argument(
        "self_energy",
        help="prefix and suffix of name of the file containing the self-energy values",
    )
    p_a.add_argument("--qe", action="store_true", help="use qe self-energy")
    p_neural = subparsers.add_parser("neural")
    args = parser.parse_args()

    qedir = QEDir(args.dir)
    E = qedir.get_energies()
    if args.orbital:
        orbitals = [args.orbital]
    else:
        orbitals = qedir.orbitals

    print(f'offset {qedir.offset} = {E[5]["DFT"]} + {0} / 2RY')

    if args.subparser == "gwwparams":
        get_model = lambda orbital: fit.fit("qe", qedir.gww_file, orbital)
    if args.subparser == "multipole":

        def get_model(orbital):
            qe_data = qedir.get_se(orbital)
            z, s = qe_data["z"], qe_data["s"]
            return fit.fit("multipole", z, s, args.n_poles)

    if args.subparser == "pade":

        def get_model(orbital):
            qe_data = qedir.get_se(orbital)
            z, s = qe_data["z"], qe_data["s"]
            M, N = range(8, 80, 8), [8, 16, 74, 76, 78, 80]
            z, s, M, N = np.array(z), np.array(s), np.array(M), np.array(N)
            models = AverageModel.get_models(z, s, M, N, precise_iter=args.precise_iter)
            w = np.imag(z)
            print(
                f"there are {np.sum([model.physical(w) for model in models])} models with physical properties"
            )
            if args.type == "avgLS":
                avgdiagmodel = AverageLSModel(models, w)
            else:
                avgdiagmodel = AverageSimilarModel(models, w)
            if True:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                avgdiagmodel.plot(ax, z, s)
                plt.show()
            return avgdiagmodel

    if args.subparser == "neural":
        import torch

        def get_model(orbital):
            qe_data = qedir.get_se(orbital)
            z, s = qe_data["z"], qe_data["s"]
            storage = StorageManager()
            model = ConvSEContX3L()
            model.load_state_dict(torch.load(storage.last_model()))
            with torch.no_grad():
                out = model(
                    torch.tensor(
                        np.stack([np.real(s), np.imag(s)]), dtype=torch.float32
                    ).unsqueeze(0)
                )
            self_on_real = out[0, 1] + 1j * out[0, 1]

            def model_fun(inpz):
                x = np.real(inpz)
                return np.interp(x, np.linspace(-20, 20, 481), self_on_real)

            return model_fun

    for i in orbitals:
        model = get_model(i)
        E[i]["GWC"] = GW_classic(E[i], model, qedir.offset)
        print(
            f"state {i}; QSEF GW energy: {np.real(E[i]['GWC']):.7}\tstarting: {np.real(E[i]['HF-pert']):.7}\tQuantum Espresso: {np.real(E[i]['GW']):.7}"
        )
        if args.plot:
            plot_se(qedir, model, i)

    # elif args.subparser == "gwwfit":
    # 	E = load_gww_energies(args.gww_out)

    # 	if args.qe:
    # 		prefix, suffix = args.self_energy.split(",")
    # 		filename_real = prefix + "-re_on_im" + suffix
    # 		filename_imag = prefix + "-im_on_im" + suffix
    # 		se_data = load_qe_se(filename_real, filename_imag)

    # 	def correlation(w):
    # 		for i in range(len(se_data["z"])-1):
    # 			if (w1 := np.imag(se_data["z"][i])) < np.real(w) and (w2 := np.imag(se_data["z"][i+1])) >= np.real(w):
    # 				return (se_data["fit_real"][i] * (w - w1) + se_data["fit_real"][i+1] * (w2 - w)) / (w2 - w1)

    # 	index = int(suffix)
    # 	offset = (E[5]["DFT"] + 0) / (2 * RY)
    # 	E0 = E[index]["DFT"]/RY - offset
    # 	E0 = complex(E0, 0)
    # 	E0 = E[index]["HF-pert"]/RY - offset + correlation(E0)

    # 	E = dyson_s(E[index]["HF-pert"] / RY - offset, E0, correlation)
    # 	E += offset
    # 	E *= RY
    # 	print(f"GW energy: {E}")
