import subprocess
import sys
import os


def run_SNR_tests():
    #SNR points to test
    snr_points = [-30,-25,-20,-15,-10,-5,0, 5, 10 , 15 , 20 , 25 , 30]

    for snr in snr_points:
        # Call the MATLAB function
        try:
            subprocess.run(["matlab", "-batch", f"SNR_tester({snr},0)"], check=True)
            print(f"MATLAB processing completed for SNR = {snr}")
        except subprocess.CalledProcessError:
            print(f"Error in MATLAB processing for SNR = {snr}")
            continue

        # Run the Python script 
        try:
            subprocess.run(["python3", "demoRNN.py"], check=True)
            print(f"Python script ran successfully for SNR = {snr}")
        except subprocess.CalledProcessError:
            print(f"Python script failed for SNR = {snr}")
            

def run_edge_tests():
    #SNR points to test
    #snr_points = [-30,-25,-20,-15,-10,-5,0, 5, 10 , 15 , 20 , 25 , 30]
    snr_points = [-10,0,10]
    for snr in snr_points:
        # Call the MATLAB function
        try:
            subprocess.run(["matlab", "-batch", f"edge_tester({snr},0,1)"], check=True)
            print(f"MATLAB processing completed for SNR = {snr}")
        except subprocess.CalledProcessError:
            print(f"Error in MATLAB processing for SNR = {snr}")
            continue

        # Run the Python script 
        try:
            subprocess.run(["python3", "demo.py"], check=True)
            print(f"Python script ran successfully for SNR = {snr}")
        except subprocess.CalledProcessError:
            print(f"Python script failed for SNR = {snr}")

def run_repmat_test():
    
    rep_points = [1 , 2, 5, 7 , 10]
    
    for rep in rep_points:
        
        try:
            subprocess.run(["matlab", "-batch", f"REPmat_tester(0,{rep})"], check=True)
            print(f"MATLAB processing completed for rep = {rep}")
        except subprocess.CalledProcessError:
            print(f"Error in MATLAB processing for rep = {rep}")
            continue

        try:
            subprocess.run(["python3", "demo.py"], check=True)
            print(f"Python script ran successfully for rep = {rep}")
        except subprocess.CalledProcessError:
            print(f"Python script failed for rep = {rep}")

def run_resample_test():
    
    res_points = [1 , 2, 5, 7 , 10]
    
    for res in res_points:
        
        try:
            subprocess.run(["matlab", "-batch", f"REPmat_tester(0,{res})"], check=True)
            print(f"MATLAB processing completed for res = {res}")
        except subprocess.CalledProcessError:
            print(f"Error in MATLAB processing for res = {res}")
            continue

        try:
            subprocess.run(["python3", "demo.py"], check=True)
            print(f"Python script ran successfully for res = {res}")
        except subprocess.CalledProcessError:
            print(f"Python script failed for res = {res}")


if __name__ == "__main__":
    #run_SNR_at_zero_frac_tests()
    run_SNR_tests()
    #run_edge_tests()
    #run_repmat_test()
    #run_resample_test()