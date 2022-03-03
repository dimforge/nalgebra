import urllib.request
import tarfile
import shutil
import os
import argparse
import time

matrix_list = {
    "pdb1HYS": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz",
    "consph": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz",
    "cant": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz",
    "pwtk": "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz",
    "rma10": "https://suitesparse-collection-website.herokuapp.com/MM/Bova/rma10.tar.gz",
    "conf5_4-8x8-05": "https://suitesparse-collection-website.herokuapp.com/MM/QCD/conf5_4-8x8-05.tar.gz",
    "shipsec1": "https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz",
    "mac_econ_fwd500": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz",
    "mc2depi": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mc2depi.tar.gz",
    "cop20k_A": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz",
    "scircuit": "https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz",
    "webbase-1M": "https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz",
    "rail4284": "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/rail4284.tar.gz"
}
tar_postfix = ".tar.gz"
matrix_postfix = '.mtx'


def download():
    for name, url in matrix_list.items():

        # Download the matrix
        print("Downloading: ", name)
        with urllib.request.urlopen(url) as response, open(name + tar_postfix, 'wb') as matrix_tarfile:
            data = response.read()
            matrix_tarfile.write(data)
        print("Finished Downloading: ", name)

        #  Extract the matrix
        print("Extracting: ", name)
        with tarfile.open(name + tar_postfix, "r:gz") as tar:
            tar.extractall()
        print("Finished Extracting: ", name)

        # Move the matrix & remove unnecessary files
        print("Cleaning Up: ", name)
        matrix_name = name + matrix_postfix
        old_path = os.path.join('.', name, matrix_name)
        new_path = os.path.join('.', matrix_name)
        shutil.move(old_path, new_path)
        shutil.rmtree(name)
        os.remove(name + tar_postfix)
        print("Finished Cleaning Up: ", name)

        # avoid being blocked from the website
        time.sleep(0.2)


def clean():
    for name in matrix_list.keys():
        filename = name + matrix_postfix
        if not os.path.isfile(filename):
            continue
        os.remove(filename)
        print("Removed: " + filename)


help_message = "you can use \'python data.py download\' or \'python data.py\' to download the matrices, \
                or \'python data.py clean\' to remove all downloaded matrices."

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("option", nargs='?', help=help_message)
    args = parser.parse_args()
    if not args.option or args.option == "download":
        download()
    elif args.option == "clean":
        clean()
    else:
        print("unknown option")