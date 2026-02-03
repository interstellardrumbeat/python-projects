import numpy as np
from pathlib import Path

def calc_ch4_ACS_from_pTcoeffs(infofile_name):
    """
    infofile_name : name of info file (same folder as working directory); text file containing:
          - path to .npz file with polynomial coefficients
          - desired pressure (p)
          - desired temperature (T)
          - output path for calculated ACS
          - control word (i.e. 'stop')

        Each of these blocks can be repeated for multiple calculations at various P an T.
        The control word must be then inserted after the last block or after the last desired one
    """

    info_file = Path(infofile_name)

    with open(info_file, 'r') as f:
        while True:
            pTcoeff_file = Path(f.readline().strip().split("\\")[-1])
            if not pTcoeff_file:
                break

            p = float(f.readline().strip())
            T = float(f.readline().strip())
            ACS_output = Path(f.readline().strip())
            controlword = f.readline().strip()

            print(f"\nProcessing: {pTcoeff_file}")
            print(f"  Pressure = {p} mbar, Temperature = {T} K")

            p_log = np.log(p + 5.0)

            data = np.load(pTcoeff_file)
            sigma = data["sigma"]
            fitted_pTcoeffs = data["fitted_pTcoeffs"]

            # Get shape
            T_poly_degree, p_poly_degree, npoints = (
                fitted_pTcoeffs.shape[0],
                fitted_pTcoeffs.shape[1],
                fitted_pTcoeffs.shape[2],
            )

            # Compute new ACS
            new_ACS = np.zeros(npoints)
            for p_index in range(p_poly_degree):
                for T_index in range(T_poly_degree):
                    new_ACS += (
                        fitted_pTcoeffs[T_index, p_index, :]
                        * (T ** T_index)
                        * (p_log ** p_index)
                    )

            # Save results
            with open(ACS_output, "w") as out:
                for s, a in zip(sigma, new_ACS):
                    out.write(f"{s:12.6f}{a * 6e-21:12.4e}\n") # Multiplication by 6e-21 added due to scaling of spectra used as input (ACS has very low magnitude; such step is required to avoid near zero values)

            print(f"New ACS written to: {ACS_output}")

            if controlword.strip().lower() == "stop":
                print("\nReached 'stop'. Exiting.")
                break

    print("\nAll calculations completed. \n")

if __name__ == '__main__':
    infofile_name = "new_ACS_fit_info_file.txt"
    calc_ch4_ACS_from_pTcoeffs(infofile_name)
