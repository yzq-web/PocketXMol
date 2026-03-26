import os
import requests

def download_single_pdb(pdb_id, save_dir):

    pdb_id = pdb_id.lower()

    # Download PDB format
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{save_dir}/{pdb_id}.pdb", 'wb') as f:
            f.write(response.content)
        print(f"{pdb_id} downloaded successfully")

    # Download cif format
    else:
        print(f"Warning: couldn't find {pdb_id} in PDB format, try to download mmCIF")
        url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
        res_cif = requests.get(url_cif)
        if res_cif.status_code == 200:
            with open(f"{save_dir}/{pdb_id}.cif", 'wb') as f:
                f.write(res_cif.content)
            print(f"{pdb_id} (mmCIF) downloaded successfully")
        else:
            print(f"Failed: {pdb_id} doesn't exist")

def batch_download_pdb(pdb_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for pdb_id in pdb_list:
        download_single_pdb(pdb_id, save_dir)


if __name__ == "__main__":
    bpep_list =  ['8cqy', '8ael', '8pdp', '8x3s', '8ia5', '7udj', '8bva', '8bq3', '8kdx', '7yue', 
                    '8q5p', '8dzv', '8wls', '8hlo', '9f42', '8ui6', '8gji', '8hlw', '7tmy', '7z6f', 
                    '8u9o', '8c1j', '8pp0', '7yod', '8h7k', '7ui8', '8wi2', '7umi', '8afr', '8he0', 
                    '7qwv', '8heq', '8e1d', '7uw2', '8ens', '8her', '8iqm', '8wx5', '8iy6', '7xv4', 
                    '8hqt', '8are', '8pii', '8f8m', '7sxh', '8t32', '8q7i', '8t8r', '8wi5', '8wxt', 
                    '8u2y', '7yoh', '8u51', '8vju', '8ck5', '8b0p', '8ti6', '8hz8', '7xuv', '8g4y', 
                    '7prx', '8wxw', '8in0', '8ahs', '8fub', '9ild', '7ue2', '8uh1', '8t5e', '8t8g', 
                    '7z6u', '8tg8', '8cd3', '9c66', '8pxx', '8sou', '8dk4', '8wqd', '8ttg', '8r10', 
                    '8exm', '8tgp', '7xfg', '8hmx', '8bia', '8s9i', '8b9t', '7z7c', '8fk3', '8c2p', 
                    '8ahy', '8sr6', '8fg6', '8jpf', '8by5', '8ttt', '8exk', '8ym2', '7xv0', '8h4x', 
                    '8igc', '7sxf', '8qu9', '8op0', '7yol', '7wul', '8opi', '8jjv', '8dvl', '8brh', 
                    '8raj', '7udk', '7trl', '8wms', '8gjg', '8gl7', '9atn', '8tgf', '8okf', '8ese', 
                    '8qfz', '8hep', '8exj', '7wqq', '8gtx', '8eq5', '8ttd', '7zw4', '8qci', '8j5u', 
                    '8gqa', '8bft', '8tte', '8gcw']
    cpep_list = ['8cv6', '8gqa', '7qs6', '8bss', '8ei4', '7ya5', '7yuz', '7zed', '7zrt', '7y8d', 
                    '8cix', '8alx', '8q1q', '8ebk', '8onu', '8ei2', '8iy6', '8f4b', '7y90', '8gjs', 
                    '8f0z', '7y99', '7zax', '8cv5', '8ibo', '8ei0', '8wky', '7z8o', '8c17', '8dvl', 
                    '8wtw', '8wm0', '8g4y', '8ei8', '8qfz']
    
    # 40 pdb id from pepbdb
    peptest_list = ['4xyn', '3vvr', '2pv3', '5a2j', '3mmg', '6f9i', '4txr', '4xhv', '5vb9', '2orz', 
                    '6drt', '3n9o', '5e2a', '4jjq', '6ddf', '3cfs', '4yv9', '3sgm', '6ie6', '2xu7', 
                    '3agy', '1k9r', '4m9z', '1rrv', '2m0u', '6f2r', '6om2', '2c9t', '4nms', '2khh', 
                    '1b8q', '1q3p', '1cf0', '6dqq', '1m21', '3ap1', '5uwh', '5ou8', '2pv2', '1xkh']
    
    bpep_path = 'data_train/bpep/files/pdbs'
    print('Processing Pep-Pro PDB:')
    batch_download_pdb(bpep_list, bpep_path)

    cpep_path = 'data_train/cpep/files/pdbs'
    print('Processing Cyclic pep-Pro PDB:')
    batch_download_pdb(cpep_list, cpep_path)

    peptest_path = 'data_train/peptest/files/pdbs'
    print('Processing PepBDB pep-Pro PDB:')
    batch_download_pdb(peptest_list, peptest_path)
