
import pickle
import os
import sys
import scipy.sparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# LOADING DATA

def load_alevin(library_names, input_path):
    '''
    Mirrors the functionality of load_inDrops (see below)

    Imports data files generated by Salmon-Alevin, when run with the --dumpMtx option. Specifically, this 
    function will expect files at the following locations:
    /input_path/library_name/alevin/quants_mat.mtx.gz
    /input_path/library_name/alevin/quants_mat_rows.txt
    /input_path/library_name/alevin/quants_mat_cols.txt
    where 'library_names' contains one or more inDrops.py output folders located at the indicated path.
    '''
    
    # Create a dictionary to hold data
    D = {}
    for j, s in enumerate(library_names):
        D[s] = {}

    # Load counts data, metadata, & convert to AnnData objects
    for s in library_names:
        
        # Load counts, gene names into AnnData structure
        D[s] = sc.read_mtx(input_path + '/' + s + '/alevin/quants_mat.mtx.gz', dtype='float32')
        D[s].var_names = np.loadtxt(input_path + '/' + s + '/alevin/quants_mat_cols.txt', dtype='str')
        D[s].obs['library_id'] = np.tile(s, [D[s].n_obs, 1])
        D[s].uns['library_id'] = s

        # Load cell barcodes into AnnData structure
        cell_bcds = np.loadtxt(input_path + '/' + s + '/alevin/quants_mat_rows.txt', dtype='str')
        
        # Append library name to each cell barcode to create unique cell IDs
        lib_cell_bcds = []
        for bcd in cell_bcds:
            lib_cell_bcds.append(s + '_' + bcd)
        D[s].obs['unique_cell_id'] = lib_cell_bcds

    return D


def load_inDrops(library_names, input_path):
    '''
    Imports data files generated by inDrops.py (https://github.com/indrops).  Specifically, this function
    will expect files at the following locations:
    /input_path/library_name/library_name.counts.tsv.gz
    /input_path/library_name/abundant_barcodes.pickle
    where 'library_names' contains one or more inDrops.py output folders located at the indicated path.
    
    The first time this function is executed, it will load counts matrices, gene names, cell names, and 
    cell barcode sequences from original tsv and pickle files, respectively.  Fast-loading versions of 
    these objects (e.g. *.npz) will be saved in place for future calls to this function.
    
    The returned dictionary object D with a ScanPy AnnData object for each library loaded, as follows:
    D[library_name] = AnnData object  
    Cell names and barcodes are stored in the adata.obs (cell barcodes as adata.obs['unique_cell_id'])
    Gene names are stored in adata.var
    Raw counts data are stored in adata.X

    This workflow allows each original library to be examined and pre-processed independently (e.g. barcode 
    filtering) prior to merging and further analysis.

    '''

    # Create a dictionary to hold data
    D = {}
    for j, s in enumerate(library_names):
        D[s] = {}

    # Load counts data, metadata, & convert to AnnData objects
    for s in library_names:
        print('_________________', s)

        # First attempt to load matrix data from preprocessed files (fast)
        if os.path.isfile(input_path + s + '/' + s + '.raw_counts.unfiltered.npz'):
            print('Loading from npz file')
            E = scipy.sparse.load_npz(
                input_path + s + '/' + s + '.raw_counts.unfiltered.npz')
            gene_names = np.loadtxt(
                fname=input_path + s + '/gene_names.txt', dtype='str')
            cell_names = np.loadtxt(
                fname=input_path + s + '/cell_names.txt', dtype='str')
            cell_bc_seqs = np.loadtxt(
                fname=input_path + s + '/cell_bc_seqs.txt', dtype='str')

        # Otherwise, load and preprocess from the original text files (slow)
        else:
            print('Loading from text file')
            counts_mat = pd.read_csv(
                input_path + s + '/' + s + '.counts.tsv.gz', sep='\t', index_col=0)
            E = scipy.sparse.coo_matrix(np.asmatrix(counts_mat.values)).tocsc()
            cell_names = counts_mat.index
            gene_names = counts_mat.columns

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
            scipy.sparse.save_npz(input_path + s + '/' +
                                  s + '.raw_counts.unfiltered.npz', E)
            np.savetxt(input_path + s + '/gene_names.txt',
                       counts_mat.columns, fmt='%s')
            np.savetxt(input_path + s + '/cell_names.txt',
                       counts_mat.index, fmt='%s')
            np.savetxt(input_path + s + '/cell_bc_seqs.txt',
                       bcd_seqs, fmt='%s')

        # Print matrix dimensions to screen
        print(E.shape, '\n')

        # Convert to ScanPy AnnData objects
        D[s] = sc.AnnData(E)
        D[s].var_names = gene_names
        D[s].obs['unique_cell_id'] = cell_bc_seqs
        D[s].obs['cell_names'] = cell_names
        D[s].obs['library_id'] = np.tile(s, [D[s].n_obs, 1])
        D[s].uns['library_id'] = s

    return D

load_inDrops_V3 = load_inDrops # alias function name 


def load_genedata(adata, csv_filename):
    '''
    Adds annotations to the 'var' dataframe of a ScanPy AnnData object (adata) from an imported CSV file.  
    Uses a set of unique identifiers (e.g. Ensembl gene IDs) to match genes.  These  identifiers must be present 
    in AnnData (in adata.obs.var_names) and in the first column of the CSV file.
    
    The structure of the CSV file is as follows:
    Column 1: unique gene identifiers (exact string matches to elements of adata.var_names)
    Column 2: first gene annotation
    Column 3: second gene annotation
      ...          ....   
    Column n: last cell annotation  
    Column headers in the CSV file (required) will become headers of new columns in adata.var  

    Unique gene ids in adata that do not appear in the CSV file will be populated with the original unique ID.
    '''
    # load the unique gene IDs from adata that will be matched to the csv file
    uID_query = adata.var_names
    
    # load CSV header, get the names and number of IDs
    header = pd.read_csv(csv_filename, nrows=0)
    annotation_names = list(header.columns.values)[
        1:]  # ignore the first column header
    nAnnotations = len(annotation_names)
    
    # make a dictionary of unique gene IDs and annotations from the CSV file
    loadtxt = np.loadtxt(csv_filename, dtype='str', delimiter=',', skiprows=1)
    annotation_dict = {}
    for uID, *annots in loadtxt:   # column1 = uID, all remaining columns are annotations
        uID=uID.replace('-','')
        annotation_dict[uID] = annots
    
    # lookup each query in the dictionary, return matching annotations (or original uID)
    annotations = []
    for j, uID in enumerate(uID_query):
        if uID in annotation_dict:
            match = annotation_dict.get(uID)
            annotations.append(match)
        else:
            annotations.append(np.repeat(uID, nAnnotations).tolist())
    
    # convert from list of lists to array
    annotations = np.array(annotations)

    # now copy the matched annotations to adata
    for j in range(0, nAnnotations):
        adata.var[annotation_names[j]] = annotations[:, j]

    return adata


def load_celldata(adata, csv_filename, filter_nomatch=False):
    '''
    Adds annotations to the 'obs' dataframe of a ScanPy AnnData object (adata) from an imported CSV file.  
    Uses a set of unique cell identifiers (e.g. inDrops cell barcode sequences) to match cells.  These 
    identifiers must be present in AnnData (as adata.obs.unique_cell_id) and in the first column of the CSV file.

    The structure of the CSV file is as follows:
    Column 1: unique cell identifiers (exact string matches to elements of adata.obs.unique_cell_id)
    Column 2: first cell annotation
    Column 3: second cell annotation
      ...          ....   
    Column n: last cell annotation  
    Column headers in the CSV file (required) will become headers of new columns in adata.obs       

    Unique cell ids in adata that no not appear in the CSV file will be annotated as 'no match'.
    'filter_nomatch' gives an option to filter these cells from the outputted version of adata.
    '''
    
    # load the unique cell IDs from adata that will be matched to the csv file
    uID_query = adata.obs.unique_cell_id
    uID_query.replace('-','')
    
    # load CSV header, get the names and number of IDs
    header = pd.read_csv(csv_filename, nrows=0)
    annotation_names = list(header.columns.values)[
        1:]  # ignore the first column header
    nAnnotations = len(annotation_names)
    
    # make a dictionary of unique cell IDs and annotations from the CSV file
    loadtxt = np.loadtxt(csv_filename, dtype='str', delimiter=',', skiprows=1)
    annotation_dict = {}
    for uID, *annots in loadtxt:   # column1 = uID, all remaining columns are annotations
        uID=uID.replace('-','')
        annotation_dict[uID] = annots
    
    # lookup each query in the dictionary, return matching annotations (or NaNs)
    annotations = []
    for j, uID in enumerate(uID_query):
        if uID in annotation_dict:
            match = annotation_dict.get(uID)
            annotations.append(match)
        else:
            annotations.append(np.repeat('no match', nAnnotations).tolist())
    
    # convert from list of lists to array
    annotations = np.array(annotations)

    # now copy the matched annotations to adata
    for j in range(0, nAnnotations):
        adata.obs[annotation_names[j]] = annotations[:, j]

    # if invoked, remove cells that were not present in the annotation CSV file
    if filter_nomatch:
        adata = adata[adata.obs[annotation_names[j]] != 'no match', :]

    return adata


# DATA PRE-PROCESSING

def filter_abundant_barcodes(adata, filter_cells=False, logscale=True, threshold=1000, library_name='', save_path='./figures/'):
    '''
    Plots a weighted histogram of transcripts per cell barcode for guiding the
    placement of a filtering threshold. Returns a filtered version of adata.  
    '''

    # If necessary, create the output directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Sum total UMI counts and genes for each cell-barcode, save to obs
    counts = np.array(adata.X.sum(1))
    genes = np.array(adata.X.astype(bool).sum(axis=1))
    adata.obs['n_counts'] = counts
    adata.obs['n_genes'] = genes

    # Plot and format a weighted cell-barcode counts histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if logscale:
        ax.hist(counts, bins=np.logspace(0, 6, 100), weights=counts / sum(counts))
    else:
        ax.hist(counts, bins=100, weights=counts / sum(counts))
    ax.set_xscale('log')
    ax.set_xlabel('Transcripts per cell barcode')
    ax.set_ylabel('Fraction of total transcripts')
    ax.set_title(library_name + ' (Weighted)')

    # Overlay the counts threshold as a vertical line
    ax.plot([threshold, threshold], ax.get_ylim())

    # Save figure to file
    fig.tight_layout()
    plt.savefig(save_path + 'barcode_hist_' + library_name + '.png')
    plt.show()
    plt.close()

    # Print the number of cell barcodes that will be retained vs. the total number of
    # cell barcodes in the library
    ix = counts >= threshold
    print('Filtering barcodes for', library_name,
          ' (', np.sum(ix), '/', counts.shape[0], ')')

    # Return a filtered version of adata
    if filter_cells:
        sc.pp.filter_cells(adata, min_counts=threshold, inplace=True)

    return adata


# VARIABLE GENES

def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    '''
    Calculate v-score (above-Poisson noise statistic) for genes in the input counts matrix
    Return v-scores and other stats
    '''

    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    def gLog(input): return np.log(input[1] * np.exp(-input[0]) + input[2])
    h, b = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))

    def errFun(b2): return np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func=errFun, x0=[b0], disp=False)
    a = c / (1 + b) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b) + b * mu_gene)
    CV_eff = np.sqrt((1 + a) * (1 + b) - 1)
    CV_input = np.sqrt(b)

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b


def runningquantile(x, y, p, nBins):
    """ calculate the quantile of y in bins of x """

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def filter_variable_genes(E, base_ix=[], min_vscore_pctl=85, min_counts=3, min_cells=3, show_vscore_plot=False, sample_name=''):
    ''' 
    Filter genes by expression level and variability
    Return list of filtered gene indices
    '''

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(
        E[base_ix, :])
    ix2 = Vscores > 0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    mu_gene = mu_gene[ix2]
    FF_gene = FF_gene[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = (((E[:, gene_ix] >= min_counts).sum(0).A.squeeze()
           >= min_cells) & (Vscores >= min_vscore))

    if show_vscore_plot:
        import matplotlib.pyplot as plt
        x_min = 0.5 * np.min(mu_gene)
        x_max = 2 * np.max(mu_gene)
        xTh = x_min * np.exp(np.log(x_max / x_min) * np.linspace(0, 1, 100))
        yTh = (1 + a) * (1 + b) + b * xTh
        plt.figure(figsize=(6, 6))
        plt.scatter(np.log10(mu_gene), np.log10(FF_gene),
                    c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
        plt.scatter(np.log10(mu_gene)[ix], np.log10(FF_gene)[
                    ix], c=np.array(['black']), alpha=0.3, edgecolors=None, s=4)
        plt.plot(np.log10(xTh), np.log10(yTh))
        plt.title(sample_name)
        plt.xlabel('Mean Transcripts Per Cell (log10)')
        plt.ylabel('Gene Fano Factor (log10)')
        plt.show()

    return gene_ix[ix]


def filter_covarying_genes(E, gene_ix, minimum_correlation=0.2, show_hist=False, sample_name=''):

    import sklearn
    import numpy as np 

    # subset input matrix to gene_ix
    E = E[:,gene_ix]
    
    # compute gene-gene correlation distance matrix (1-correlation)
    #gene_correlation_matrix1 = sklearn.metrics.pairwise_distances(E.todense().T, metric='correlation',n_jobs=-1)
    gene_correlation_matrix = 1-sparse_corr(E) # approx. 2X faster than sklearn
  
    # for each gene, get correlation to the nearest gene neighbor (ignoring self)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr = 1-gene_correlation_matrix.min(axis=1)
  
    # filter genes whose nearest neighbor correlation is above threshold 
    ix_keep = np.array(max_neighbor_corr > minimum_correlation, dtype=bool).squeeze()
  
    # plot distribution of top gene-gene correlations
    if show_hist:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.hist(max_neighbor_corr,bins=100)
        plt.title(sample_name)
        plt.xlabel('Nearest Gene Correlation')
        plt.ylabel('Counts')
        plt.show()
  
    return gene_ix[ix_keep]


# GEPHI IMPORT & EXPORT

def export_to_graphml(adata, filename='test.graphml', directed=None):    
    import igraph as ig

    adjacency = adata.uns['neighbors']['connectivities']

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shap[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warn('The constructed graph has only {} nodes. '
                  'Your adjacency matrix contained redundant nodes.'
                  .format(g.vcount()))
    g.write_graphml(filename)


def import_pajek_xy(adata, filename='test.net'):
    
    # first determine the number of graph nodes in *.net file
    with open(filename,'r') as file:
        nNodes = 0
        for ln,line in enumerate(file):
            if line.startswith("*Edges"):
                nNodes = ln-1

    # extract xy coordinates from *.net file
    with open(filename,'r') as file:
        lines=file.readlines()[1:nNodes+1] 
        xy = np.empty((nNodes,2))
        for ln,line in enumerate(lines):
            xy[ln,0]=(float(line.split(' ')[2]))
            xy[ln,1]=(float(line.split(' ')[3]))

    # generate ForceAtlas2 data structures and update coordinates
    sc.tl.draw_graph(adata, layout='fa', iterations=1)
    adata.obsm['X_draw_graph_fa']=xy

    return adata


# CLASSIFICATION

def train_classifiers(X, labels, PCs, gene_ind):
    '''
    Trains a series of machine learning classifiers to associate individual cells with class labels.
    Does so in a low-dimensional PCA representation of the data (PCs) over pre-defined genes (gene_ind).
    '''

    # Import sklearn classifier packages
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    


    # Subset by gene indices; project X into PCA subspace
    X_ind = X[:,gene_ind]
    PCs_ind = PCs[gene_ind,:]
    X_PCA = np.matmul(X_ind,PCs_ind)
    
    # Specify classifiers and their settings 
    classifier_names = ['NearestNeighbors', 'SVM-Linear', 'SVM-RBF', 'DecisionTree', 'RandomForest', 
                        'NeuralNet', 'NaiveBayes', 'LDA']
    classifiers = [KNeighborsClassifier(20, weights='distance', metric='correlation'),
                   SVC(kernel='linear', gamma='scale', C=1, random_state=802),
                   SVC(kernel='rbf', gamma='scale', C=1, random_state=802),
                   DecisionTreeClassifier(random_state=802),
                   RandomForestClassifier(n_estimators=200, random_state=802),
                   MLPClassifier(random_state=802),
                   GaussianNB(),
                   LinearDiscriminantAnalysis()]
    
    # Split data into training and test subsets
    X_train, X_test, labels_train, labels_test = train_test_split(X_PCA, labels, test_size=0.5, random_state=802)
        
    # Build a dictionary of classifiers
    scores = []
    ClassifierDict={}
    for n,name in enumerate(classifier_names):
        clf_test = classifiers[n].fit(X_train, labels_train)
        score = clf_test.score(X_test, labels_test)
        scores.append(score)
        print(name,round(score,3))
        ClassifierDict[name]=classifiers[n].fit(X_PCA, labels)
    
    # Export classifier dictionary and subspace projection objects

    return {'Classes' : np.unique(labels),
            'Classifiers' : ClassifierDict,
    		'Classifier_Scores' : dict(zip(classifier_names, scores)), 
            'PC_Loadings' : PCs,
            'Gene_Ind' : gene_ind}
   

def predict_classes(adata, Classifier):    
    '''
    '''
    X = adata.X
    X[np.isnan(X)]=0
    PCs = Classifier['PC_Loadings']
    gene_ind = Classifier['Gene_Ind']

    # First check to see if genes match between adata and Classifier 
    adata_genes = np.array(adata.var.index) 
    classifier_genes = np.array(gene_ind.index)
    if len(classifier_genes)==len(adata_genes):
        if (classifier_genes==adata_genes).all():
            # Subset by gene indices; project X into PCA subspace
            X_ind = X[:,gene_ind]
            PCs_ind = PCs[gene_ind,:]
            X_PCA = np.matmul(X_ind,PCs_ind)
    
    else:
        # Match highly variable classifier genes to adata genes, correcting for case
        adata_genes = np.array([x.upper() for x in adata_genes])
        classifier_genes = np.array([x.upper() for x in np.array(classifier_genes[gene_ind])])
        # Get overlap
        gene_overlap, dataset_ind, classifier_ind = np.intersect1d(adata_genes,classifier_genes,return_indices=True)
        # Subset by gene indices; project X into PCA subspace
        PCs_ind = PCs[gene_ind,:]
        PCs_ind = PCs_ind[classifier_ind,:]
        X_ind = X[:,dataset_ind]
        X_PCA = np.matmul(X_ind,PCs_ind)

    # Predict class labels and probabilities for each cell, store results in adata
    for n,name in enumerate(Classifier['Classifiers']):
        adata.obs['pr_'+name] = Classifier['Classifiers'][name].predict(X_PCA)
        if hasattr(Classifier['Classifiers'][name], "predict_proba"): 
            adata.obsm['proba_'+name] = Classifier['Classifiers'][name].predict_proba(X_PCA)

    return adata


# CLUSTERING
    
def plot_confusion_matrix(labels_A, labels_B,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          overlay_values=False,
                          vmin=None,
                          vmax=None):
    '''
    Plots a confusion matrix comparing two sets labels. 

    '''

    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    # Compute confusion matrix; 
    cm = confusion_matrix(labels_A, labels_B)
    non_empty_rows = cm.sum(axis=0)!=0
    non_empty_cols = cm.sum(axis=1)!=0
    cm = cm[:,non_empty_rows]
    cm = cm[non_empty_cols,:]
    cm = cm.T

    # Classes are the unique labels
    classes = np.unique(labels_A.append(labels_B))

    # Normalize by rows (label B)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Set title, colorbar, and axis names
    if normalize:
        colorbar_label = 'Fraction Overlap'
        if not title:
            title = 'Normalized confusion matrix'
    else:
        colorbar_label = '# Overlaps'
        if not title:
        	title = 'Confusion matrix, without normalization'  
  
    if hasattr(labels_A, 'name'):
        labels_A_name = labels_A.name #.capitalize()   	
    else:
        labels_A_name = 'Label A'
    if hasattr(labels_B, 'name'):
        labels_B_name = labels_B.name #.capitalize()    	
    else:
        labels_B_name = 'Label B'

    # Generate and format figure axes
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.grid(False)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes[non_empty_cols], yticklabels=classes[non_empty_rows],
           title=title,
           ylabel=labels_B_name,
           xlabel=labels_A_name)

    # Format tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va='top',
             rotation_mode='anchor',fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    # Format colorbar
    cb=ax.figure.colorbar(im, ax=ax, shrink=0.5)
    cb.ax.tick_params(labelsize=10) 
    cb.ax.set_ylabel(colorbar_label, rotation=90)
    
    # Loop over data dimensions and create text annotations
    if overlay_values:
        fmt = '.1f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        size=8)
    ax.set_aspect('equal') 
    
    return fig, ax


# PCA

def pca_heatmap(adata, component, use_raw=None, layer=None):
    attr = 'varm'
    keys = 'PCs'
    scores = getattr(adata, attr)[keys][:, component]
    dd = pd.DataFrame(scores, index=adata.var_names)
    var_names_pos = dd.sort_values(0, ascending=False).index[:20]

    var_names_neg = dd.sort_values(0, ascending=True).index[:20]

    pd2 = pd.DataFrame(adata.obsm['X_pca'][:, component], index=adata.obs.index)

    bottom_cells = pd2.sort_values(0).index[:300].tolist()
    top_cells = pd2.sort_values(0, ascending=False).index[:300].tolist()

    sc.pl.heatmap(adata[top_cells+bottom_cells], list(var_names_pos) + list(var_names_neg), 
                        show_gene_labels=False,
                        swap_axes=True, cmap='viridis', 
                        use_raw=False, layer=layer, vmin=-1, vmax=3, figsize=(3,3))
                        

# DIFFERENTIAL EXPRESSION

def get_dynamic_genes(adata, sliding_window=100, fdr_alpha = 0.05, min_cells=20, nVarGenes=2000):

    # Input an AnnData object that has already been subsetted to cells and (optionally) genes of interest.
    # Cells are ranked by dpt pseudotime. Genes are tested for significant differential expression 
    # between two sliding windows corresponding the highest and lowest average expression. FDR values
    # are then calculated by thresholding p-values calculated from randomized data.
    # Returns a copy of adata with the following fields added: 
    #   adata.var['dyn_peak_cell']: pseudotime-ordered cell with the highest mean expression
    #   adata.var['dyn_fdr']: fdr-corrected p-value for differential expression
    #   adata.var['dyn_fdr_flag']: boolean flag, true if fdr <= fdr_alpha

    import scipy.stats

    # Function for calculating p-values for each gene from min & max sliding window expression values
    def get_slidingwind_pv(X, sliding_window):
        # construct a series of sliding windows over the cells in X
        wind=[]
        nCells = X.shape[0]
        for k in range(nCells-sliding_window+1):    
            wind.append(list(range(k, k+sliding_window)))
        # calculate p-values on the sliding windows
        pv = []
        max_cell_this_gene = []
        nGenes = X.shape[1]
        for j in range(nGenes):
            tmp_X_avg = []
            # get mean expression of gene j in each sliding window k
            for k in range(len(wind)-1):    
                tmp_X_avg.append(np.mean(X[wind[k],j]))
            # determine min and max sliding windows for this gene
            max_wind = np.argmax(tmp_X_avg)
            min_wind = np.argmin(tmp_X_avg)
            # determine if this gene displays significant differential expression
            _,p=scipy.stats.ttest_ind(X[wind[max_wind],j],X[wind[min_wind],j])
            pv.append(p[0])
            max_cell_this_gene.append(max_wind)
        return np.array(pv), np.array(max_cell_this_gene)

    # pre-filter genes based on minimum expression 
    adata.X = adata.raw.X
    expressed_genes = np.squeeze(np.asarray(np.sum(adata.X  >= 1, axis=0) >= min_cells))
    adata = adata[:,expressed_genes]
    nGenes_expressed = adata.shape[1]

    # pre-filter genes based on variability
    nVarGenes = min([nGenes_expressed, nVarGenes])
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10**6) # TPM normalization
    sc.pp.highly_variable_genes(adata, n_top_genes=nVarGenes)
    adata = adata[:,adata.var['highly_variable'] == True]
    
    # import counts and pseudotime from the AnnData object
    cell_order = np.argsort(adata.obs['dpt_pseudotime'])
    if scipy.sparse.issparse(adata.X):
        X = adata.X[cell_order,:].todense()
    else:
        X = adata.X[cell_order,:]

    # calculate p values on the pseudotime-ordered data
    print('calculating p-values')
    pv, peak_cell = get_slidingwind_pv(X, sliding_window)
    adata.var['dyn_peak_cell'] = peak_cell#np.argsort(gene_ord)
    print('done calculating p-values')
    
    # calculate p values on the randomized data
    print('calculating randomized p-values')
    np.random.seed(802)
    X_rand = X[np.random.permutation(cell_order),:]
    pv_rand, _ = get_slidingwind_pv(X_rand, sliding_window)
    print('done calculating randomized p-values')

    # calculate fdr as the fraction of randomized p-values that exceed this p-value
    print('calculating fdr')
    fdr = []
    fdr_flag = []
    nGenes = adata.shape[1]
    for j in range(nGenes):
        fdr.append(sum(pv_rand <= pv[j])/nGenes)
        fdr_flag.append(fdr[j] <= fdr_alpha)
    adata.var['dyn_fdr'] = fdr
    adata.var['dyn_fdr_flag'] = fdr_flag
    print('done calculating fdr')

    return adata

    
# PLOTTING

def format_axes(eq_aspect='all', rm_colorbar=False):
    '''
    Gets axes from the current figure and applies custom formatting options
    In general, each parameter is a list of axis indices (e.g. [0,1,2]) that will be modified
    Colorbar is assumed to be the last set of axes
    '''
    
    # get axes from current figure
    ax = plt.gcf().axes

    # format axes aspect ratio
    if eq_aspect is not 'all':
        for j in eq_aspect:
            ax[j].set_aspect('equal') 
    else:
        for j in range(len(ax)):
            ax[j].set_aspect('equal') 

    # remove colorbar
    if rm_colorbar:
        j=len(ax)-1
        if j>0:
            ax[j].remove()


# SPARSE MATRICES

def sparse_corr(A):
        
    N = A.shape[0]
    C=((A.T*A -(sum(A).T*sum(A)/N))/(N-1)).todense()
    V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
    COR = np.divide(C,V+1e-119)
    
    return COR


# TRACERSEQ ANALYSIS


def load_tracerseq_barcode_counts(adata, key, path):

  # load TracerSeq counts file into a pandas dataframe
  df = pd.read_csv(path, dtype='str', delimiter=',', header=None)
  df.columns =['unique_cell_id', 'UniqueTracerID', 'TracerBarcode', 'UMI_counts']
  df['unique_cell_id'] = df['unique_cell_id'].str.replace("A_", "A-").str.replace("T_", "T-").str.replace("G_", "G-").str.replace("C_", "C-")

  # filter the dataframe to only include cell barcodes present in adata
  cells_flag = np.in1d(df['unique_cell_id'], adata.obs['unique_cell_id'].tolist()) 
  df = df.drop(df[~cells_flag].index).reset_index(drop=True)
  df['UniqueTracerID'] = np.unique(df['UniqueTracerID'], return_inverse=True)[1] # 'reset' the UniqueTracerID column

  # create an empty counts matrix 'm': one row for each cell barcode, one column for each unique TracerSeq barcode, entries will be UMI counts
  nCells_adata = len(adata.obs.unique_cell_id)
  nUniqueTracerBarcodes = len(np.unique(df['UniqueTracerID']))
  m = np.zeros((nCells_adata,nUniqueTracerBarcodes))

  # create an empty array 'bcd': one entry for each unique TracerSeq barcode sequence
  bcd = np.array([None] * nUniqueTracerBarcodes)
  
  # populate the m matrix with UMI counts for each cell-TracerSeq barcode pair
  # populate the bcd list with the original TracerSeq barcode sequences
  for r in range(len(df)): 
      this_row = np.where(np.in1d(adata.obs['unique_cell_id'],df['unique_cell_id'][r]))[0][0]
      this_column = int(df['UniqueTracerID'][r]) - 1 # convert to zero-based index
      m[this_row,this_column] = df['UMI_counts'][r]
      bcd[this_column]=(df['TracerBarcode'][r])

  # filter to only include TracerSeq barcodes that comprise a clone (2 cells or more)
  nTracerBarcodes = m.shape[1]
  clones_flag = np.count_nonzero(m, axis=0)>1
  nClones = np.count_nonzero(clones_flag)
  while nTracerBarcodes > nClones:
    m = m[:,clones_flag]
    bcd = bcd[clones_flag]
    nTracerBarcodes = m.shape[1]
    clones_flag = np.count_nonzero(m, axis=0)>1
    nClones = np.count_nonzero(clones_flag)

  print(key, 'nTracerBarcodes:', m.shape[1])
  print(key, 'nTracerCells:', np.count_nonzero(~np.all(m == 0, axis=1)))

  # export to 'TracerSeq' adata.obsm dataframe
  df_export = pd.DataFrame(data = m, index = adata.obs.index.copy(), columns = [key + "_" + bcd])
  if 'TracerSeq' in list(adata.obsm.keys()): # if 'TracerSeq' obsm already exists, append to it
    adata.obsm['TracerSeq'] = pd.concat([adata.obsm['TracerSeq'], df_export], axis = 1)
  else:
    adata.obsm['TracerSeq'] = df_export
  
  # drop duplicate columns, if present
  adata.obsm['TracerSeq'] = adata.obsm['TracerSeq'].T.drop_duplicates().T

  return adata


def plot_cells_vs_barcodes_heatmap(adata, cell_labels_key=None, umi_thresh=0):

  import seaborn as sns
  import sys
  
  X = adata.obsm['TracerSeq']

  # convert TracerSeq counts matrix to boolean based on UMI threshold
  X = (X > umi_thresh)*1
  
  # filter cells with both transcriptome and TracerSeq information
  flag = X.sum(axis = 1) > 0
  X = X[flag]

  # plot a clustered heatmap of cells x barcodes 
  sys.setrecursionlimit(100000) 
  
  # set up cell labels
  if cell_labels_key is not None:
    cell_labels = adata.obs[cell_labels_key]
    cell_label_colors = adata.uns[cell_labels_key + '_colors']
    lut=dict(zip(np.unique(cell_labels),cell_label_colors))
    row_colors = cell_labels.map(lut)
    row_colors = row_colors[flag]
  else:
    row_colors=[]
  
  # generate cluster map with or without cell labels
  if cell_labels_key is not None:
    cg = sns.clustermap(X, 
                        metric='jaccard', cmap='Greys', 
                        cbar_pos=None, 
                        xticklabels=False, yticklabels=False,
                        dendrogram_ratio=0.08, figsize=(6, 8),
                        row_colors=row_colors,
                        colors_ratio=0.02)
  else:
    cg = sns.clustermap(X, 
                    metric='jaccard', cmap='Greys', 
                    cbar_pos=None, 
                    xticklabels=False, yticklabels=False,
                    dendrogram_ratio=0.08, figsize=(6, 8))
  
  # format plot
  cg.ax_heatmap.set_xlabel('Clones')
  cg.ax_heatmap.set_ylabel('Cells')
  for _, spine in cg.ax_heatmap.spines.items():
    spine.set_visible(True) # draws a simple frame around the heatmap
  cg.ax_col_dendrogram.set_visible(False) # hide the column dendrogram 


def plot_state_couplings_heatmap(X, state_IDs=None, title=None, tick_fontsize=10, figsize=8, do_clustering=False):   
    
    # Plot a Seaborn clustermap of state-state barcode couplings

    if state_IDs is not None:
      X = pd.DataFrame(X, index=state_IDs, columns=state_IDs)
    
    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    cg = sns.clustermap(X, metric='correlation', method='average', cmap='viridis', 
                        cbar_pos=None, dendrogram_ratio=0.2, figsize=(figsize,figsize),
                        col_cluster = do_clustering, row_cluster = do_clustering,
                        xticklabels = 1, yticklabels = 1,colors_ratio=0.02, vmax=vmax)  
    cg.ax_col_dendrogram.set_visible(False) # hide the column dendrogram
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize = tick_fontsize)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize = tick_fontsize)
    plt.title(title)


def get_observed_barcode_couplings(adata, cell_state_key, umi_thresh=0, thresh_min_cells_per_hit=1):
  
  # Calculate 'OBSERVED' barcode couplings between states

  # For all state pairs, sum the number of times a cell with a given TracerSeq barcode hit both state j and state k

  # import data
  adata = adata[~adata.obs['CellTypeName'].isin(['NaN']),:]
  X = adata.obsm['TracerSeq']
  cell_states = adata.obs[cell_state_key]
  
  # convert TracerSeq counts matrix to boolean based on UMI threshold
  X = np.array(X > umi_thresh)*1
  
  # filter to cells with both state (transcriptome) and TracerSeq information, filter out states with zero hits
  flag = X.sum(axis = 1) > 0
  X = X[flag]
  cell_states = cell_states[flag]
  coupled_state_IDs = np.unique(cell_states)
  nStates = len(coupled_state_IDs)
  
  # compute the observed couplings matrix
  X_obs = np.zeros((nStates,nStates))  
  for j in range(nStates):    
    cells_in_state_j = np.array(coupled_state_IDs[j] == cell_states) # index the cells assigned to this particular j state
    clone_hits_in_state_j = sum(X[cells_in_state_j,:]) >= thresh_min_cells_per_hit
    for k in range(j,nStates): # calculate upper triangle only to save time
      cells_in_state_k = np.array(coupled_state_IDs[k] == cell_states) # index the cells assigned to this particular k state
      clone_hits_in_state_k = sum(X[cells_in_state_k,:]) >= thresh_min_cells_per_hit    
      X_obs[j,k] = sum(clone_hits_in_state_j & clone_hits_in_state_k)
  X_obs = np.maximum(X_obs,X_obs.transpose()) # re-symmetrize the matrix

  return X_obs, coupled_state_IDs


def get_oe_barcode_couplings(X_obs):

  # Calculate 'OBSERVED/EXPECTED' barcode couplings between states

  # Given the observed barcode couplings matrix, coupling frequencies expected by random chance are the outer product of the column sums and row sums normalized by the total.

  X = np.array(X_obs)
  X_expect = X.sum(0, keepdims=True) * X.sum(1, keepdims=True) / X.sum()
  X_oe = X_obs/X_expect

  return X_oe




# FORCE LAYOUT
# CELL NORMALIZATION
# DIMENSIONALITY REDUCTION
