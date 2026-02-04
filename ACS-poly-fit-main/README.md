# Polynomial fit for Absorption Cross Sections

## Glossary:  
ACS: Absorption Cross Section  
Sigma: wavenumber ($$cm^{-1}$$)

## Description:
The main software (acs_poly_fit) takes in spectra in the form "wavenumber (cm-1) vs scaled Absorption Cross Section (ACS)"
at various temperature and pressures and fits them to a polynomial in two-dimensions, solving for coefficients "a" of:

$$f(T, p) = \displaystyle\sum_{i=0}^{T \ degree} \displaystyle\sum_{j=0}^{p \ degree} a(i,j) \ \ T^i \ \ \log(p+5)^j$$

where T is the temperature in Celsius and P the pressure in mbar. The logarithmic pressure term log(p+5) is used to stabilize the fit at low pressures (log base: natural).

The degree in T and P can be decided by the user based on the amount of data available. 
The results are then stored in a .npz file that can be used to fit new ACSs at T and P not available experimentally.
Such fit can be done using the script and info file in the folder "New ACS fit".
  
&nbsp;  
/)  /)  ~   ┏━━━━━━━━━━━┓  
( •-• )  ~    HOW TO USE  
/づづ ~  ┗━━━━━━━━━━━┛    

**This section describes the required inputs and how to structure them.**

The "acs_poly_fit" script needs two inputs:   
### 1) an info file:  
its path and filename must be provided at the end of the script, in the "main" call (info_path and info_filename);
the info file MUST have a certain structure.  
See section [Info file](#info-file) below for details. A sample is provided in this repository.

### 2) Spectra:    
the spectra must be made of only two columns, of only numbers (no header, title, column names, etc...); 
the first column must contain the wavenumber/sigma values, the second the ACSs;  
IMPORTANT: to avoid near-zero values and given the small magnitude of ACSs,
it´s recommended to multiply the ACSs by 10^20 (or close).

## Info file  
The info file read by the software has two sections and must follow this exact order:    
### 1) the general section (first 5 lines), which appears only once at the start of the file;

* **Input folder:** the URL of the folder where the spectra are stored  
* **Output folder:** the URL of the desired output folder  
* **Range for fit (in wavenumber, cm-1):**
    - single range - min max (eg. 6076.200 6077.600)  
    - multiple ranges (optional): min max min max ... (eg. 607.6 607.7 608.1 608.2 ...)    
        >NOTE: The software will parse these as independent regions. The region used for the fit is selected inside `main()` via the `region` index (default: `region = 0`, i.e. the first range).       
* **Degree of polynomial for Pressure:** an integer defining the degree of the polynomial in P (eg. 4)  
* **Degree of polynomial for Temperature:** an integer defining the degree of the polynomial in T (eg. 2)  

### 2) the per-spectrum section (7 lines).  
Each block can be repeated for every spectrum that the user wants to feed into the software:

* **ACS file:** name of the input spectrum  
* **noise:** experimental noise of the spectrum (eg. 1.0)  
* **additional weight:** additional weight to be included in the calculation for this spectrum  
    - 1: no additional weight (multiply by 1)  
* **P_tot:** Pressure in mbar (eg. 66.9325)  
* **T:** temperature in Kelvin (eg. 312.62058)  
* **Baseline polynomial degree:** degree of the polynomial to be used in the baseline correction (eg. 1, 2, ...)  
    - -1: no baseline fitted   
* **ACS scaling:** controls whether a multiplicative scaling factor is fitted , often necessary if not all spectra are measured in the same lab
    - 1: fit a scaling factor
    - any value: scaling is fixed to the given number (no fit)   

## Roadmap
- [ ] Faster algorithm - curve_fit is very slow, maybe switch to scikitlearn or numba  
- [ ] Plotting - showing the model vs observed spectra, the baseline and the residuals (plus other stuff)  
- [ ] x-axis mismatch handling - automatically interpolate to same grid and re-feed in the fit   
- [ ] Directly connect to fitted ACS calculator (for not experimentally available Ts and Ps)  
- [ ] Implement use of covariance results (in plot and ACS calculator) - currently only stored, not used    
- [ ] GUI version       

## Contributing
Open to suggestions to make it _better, faster, stronger_ (cit.).

### Authors and acknowledgment
The use of comments and docstring (tries) to follow this conventions:  
[PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)  
[PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
