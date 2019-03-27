
import pickle
import numpy as np
import os
import scipy.sparse
from scipy import optimize
import pandas as pd
import scanpy.api as sc
import matplotlib.pyplot as plt



########## GENERIC


########## LOADING DATA

def load_inDrops_V3(library_names, input_path):
    '''
    Imports inDrops V3 data files.  The first time this function is executed, it will load
    counts matrices, gene names, cell names, and cell barcode sequences from original tsv and pickle
    files, respectively.  Fast-loading versions of these objects (e.g. *.npz) will be saved for 
    future calls to this function.
    The returned dictionary object D includes the following entries: 
    'E', meta', 'gene_names', 'cell_names', 'cell_bc_seqs'
    '''

    # Create a dictionary to hold data
    D = {}
    for j,s in enumerate(library_names):
        D[s] = {}

    # Load counts data, metadata, & convert to AnnData objects
    for s in library_names:
        print('_________________', s)
        
        # First attempt to load matrix data from preprocessed files (fast)
        if os.path.isfile(input_path + s + '/' + s + '.raw_counts.unfiltered.npz'):
            print('Loading from npz file')
            E = scipy.sparse.load_npz(input_path + s + '/' + s + '.raw_counts.unfiltered.npz')
            gene_names = np.loadtxt(fname=input_path + s + '/gene_names.txt', dtype='str')
            cell_names = np.loadtxt(fname=input_path + s + '/cell_names.txt', dtype='str')    
            cell_bc_seqs = np.loadtxt(fname=input_path + s + '/cell_bc_seqs.txt', dtype='str')

        # Otherwise, load and preprocess from the original text files (slow)
        else:
            print('Loading from text file')
            counts_mat = pd.read_csv(input_path + s + '/' + s + '.counts.tsv.gz',sep='\t',index_col=0)
            E = scipy.sparse.coo_matrix(np.asmatrix(counts_mat.values)).tocsc()
            cell_names = counts_mat.index;
            gene_names = counts_mat.columns;
            
            # Load the barcode dictionary pickle file, format as keys=bcodes; values=sequences
            f = open(input_path + s + '/abundant_barcodes.pickle', 'rb')
            bc_dict = pickle.load(f)
            f.close()     
            bcd_dict = {bc_dict[bc][0]: bc for bc in bc_dict}
        
            # Get barcode sequences corresponding to each cell index
            bcd_seqs = []
            for cname in counts_mat.index:
                bcd_seqs.append(s + '_' + bcd_dict.get(cname))
            cell_bc_seqs = bcd_seqs

            # Save fast files for next time
            scipy.sparse.save_npz(input_path + s + '/' + s + '.raw_counts.unfiltered.npz', E)       
            np.savetxt(input_path + s + '/gene_names.txt', counts_mat.columns, fmt='%s')
            np.savetxt(input_path + s + '/cell_names.txt', counts_mat.index, fmt='%s')
            np.savetxt(input_path + s + '/cell_bc_seqs.txt', bcd_seqs, fmt='%s')

        # Print matrix dimensions to screen
        print(E.shape, '\n')

        # Convert to ScanPy AnnData objects
        D[s]['adata'] = sc.AnnData(E)
        D[s]['adata'].obs['n_counts'] = D[s]['adata'].X.sum(1).A1
        D[s]['adata'].var_names = gene_names 
        D[s]['adata'].obs['unique_cell_id'] = cell_bc_seqs
        D[s]['adata'].obs['cell_names'] = cell_names
        D[s]['adata'].obs['library_id'] = np.tile(s, [D[s]['adata'].n_obs,1])  
        D[s]['adata'].uns['library_id'] = s

    return D


def import_celldata(adata, csv_filename, filter_nomatch=False):
    '''
    Adds cell annotations to the 'obs' dataframe of a ScanPy AnnData object (adata) from an imported CSV file.  
    Uses a set of unique cell identifiers (e.g. inDrops cell barcode sequences) to match cells.  These 
    identifiers are present in AnnData (in adata.obs.unique_cell_id) and in the first column of the CSV file.
    
    The structure of the CSV file is as follows:
    Column 1: unique cell identifiers (exact string matches to elements of adata.obs.unique_cell_id)
    Column 2: first cell annotation
    Column 3: second cell annotation
      ...          ....   
    Column n: last cell annotation  
    Column headers in the CSV file (required) will become headers of new columns in adata.obs       
    
    Unique cell ids in adata that no not appear in the CSV file will be annotated as 'no match'.
    'filter_nomatch' gives an option to remove these cells in the outputted version of adata.
    '''
    
    uID_query = adata.obs.unique_cell_id

    # load CSV header, get the names and number of IDs
    header = pd.read_csv(csv_filename, nrows=0)
    annotation_names = list(header.columns.values)[1:] # ignore the first column header
    nAnnotations = len(annotation_names)
    
    # make a dictionary of unique cell IDs and annotations from the CSV file
    loadtxt=np.loadtxt(csv_filename,dtype='str',delimiter=',',skiprows=1)    
    annotation_dict = {}
    for uID, *annots in loadtxt:   # column1 = uID, all remaining columns are annotations
        annotation_dict[uID] = annots

    # lookup each query in the dictionary, return matching annotations (or NaNs)
    annotations=[]
    for j,uID in enumerate(uID_query):
        if uID in annotation_dict:
            match=annotation_dict.get(uID)
            annotations.append(match)
        else:
            annotations.append(np.repeat('no match',nAnnotations).tolist())

    # convert from list of lists to array
    annotations=np.array(annotations)

    # now copy the matched annotations to adata
    for j in range(0,nAnnotations):
        adata.obs[annotation_names[j]] = annotations[:,j]

    # if invoked, remove cells that were not present in the annotation CSV file
    if filter_nomatch:
        adata = adata[adata.obs[annotation_names[j]]!='no match',:]

    return adata


########## DATA PRE-PROCESSING

def filter_abundant_barcodes(adata):
    '''
    Plots a weighted histogram of transcripts per cell barcode for guiding the
    placement of a filtering threshold. Returns a filtered version of adata.  
    '''

    # Load counts data etc from adata
    counts = adata.obs['n_counts'].values
    threshold = adata.uns['counts_thresh']
    library_name = adata.uns['library_id']

    # Plot and format a weighted counts histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=np.logspace(0, 6, 100), weights=counts/sum(counts))
    ax.set_xscale('log')
    ax.set_xlabel('Transcripts per cell barcode')
    ax.set_ylabel('Fraction of total transcripts')
    ax.set_title(library_name + ' (Weighted)')
    
    # Overlay the counts threshold as a vertical line
    ax.plot([threshold,threshold],ax.get_ylim());

    # Save figure to file
    fig.tight_layout()
    plt.savefig('plots/' + library_name + '_barcode_hist.pdf')
    plt.close()

    # Print the number of cell barcodes retained vs. the total number of cell barcodes in the library
    ix = counts >= threshold
    print('Filtering barcodes for', library_name, ' (', np.sum(ix), '/', counts.shape[0], ')') 

    # return a filtered version of adata
    adata_filt = sc.pp.filter_cells(adata, min_counts=threshold, copy=True)
    return adata_filt


########## GENE OSCILLATIONS

def test_oscillation(x_data, y_data):

    # Plot raw data
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data)

    # Fit a sine function
    def test_func(x, a, b):
        return a * np.sin(b * x)
    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
    print(params)


    # Plot curve
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, test_func(x_data, params[0], params[1]),
             label='Fitted function')
    plt.legend(loc='best')
    plt.show()


########## CELL NORMALIZATION
########## DIMENSIONALITY REDUCTION
########## GRAPH CONSTRUCTION
########## FORCE LAYOUT
########## CLUSTERING
########## GENE ENRICHMENT
########## PLOTTING STUFF

