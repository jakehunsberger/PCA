import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class PCA_Full(PCA):
    #Can handle missing data and perform CV for hyperparameter optimization
    
    def __init__(self, n_components=None, max_components=None, num_folds=None):
        super().__init__(n_components=n_components)
        self.Max_Components = max_components
        self.Folds_Num = num_folds
        return
    
    #Fit with missing data
    def fit(self, X_Train):
        #Assumes data has already been mean/var standardized
        
        #Ascertain which data points are missing with boolean array
        Nulls_Bool_np = np.isnan(X_Train)
        
        #Fill missing data points with zero to begin
        X_Train[Nulls_Bool_np] = 0
        
        
        Fit_Iterations = 3
        for i in range(Fit_Iterations):
            
            #Fit PCA model with most current estimates for missing values
            #self.fit(X_Train)
            super().fit(X_Train)
            
            #Use all data to project to latent space
            #X_Latent = self.transform(X_Train)
            X_Latent = super().transform(X_Train)
            
            
            #Use latent space scores to make better predictions for missing values
            #X_Pred = self.inverse_transform(X_Latent)
            X_Pred = super().inverse_transform(X_Latent)
            
            #Update only missing data values with prediction values
            X_Train = (1 - Nulls_Bool_np) * X_Train + Nulls_Bool_np * X_Pred
            continue
        
        return
    
    
    def fit_CV(self, X_Data, Verbose=False):
        
        #Observation Indices
        Obs_Indices = np.linspace(start=0, stop=X_Data.shape[0]-1, num=X_Data.shape[0], dtype='int')
        
        #Max number of components must be less than either total observations/variables count
        Max_Components = min(self.Max_Components, X_Data.shape[0]-1, X_Data.shape[1]-1)
        
        #Loop over principal component number candidates
        Best_Q2 = -1
        for Components_Num in range(1, Max_Components+1):
            
            
            #Loop over observation folds
            Obs_Folder = KFold(n_splits=self.Folds_Num)
            np.random.shuffle(Obs_Indices)
            Obs_Fold_Q2s = []
            for Train_Group, Test_Group in Obs_Folder.split(Obs_Indices):
                
                #KFold.split() only returns indices of Obs_Indices
                #Does not actually split Obs_Indices into train/test arrays
                Train_Indices = Obs_Indices[Train_Group]
                Test_Indices = Obs_Indices[Test_Group]
                
                #Split X_Data into train/test sets
                X_Train = X_Data[Train_Indices]
                X_Test = X_Data[Test_Indices]
                
                
                #Instantiate new PCA model object
                super().__init__(n_components=Components_Num)
                
                #Fit new PCA model object to train data
                self.fit(X_Train)
                
                
                Q2 = self.score(X_Test)
                Obs_Fold_Q2s.append(Q2)
                
                continue
            Obs_Fold_Q2s = np.array(Obs_Fold_Q2s)
            Curr_Q2 = np.mean(Obs_Fold_Q2s)
            
            #Check for more optimal score from components number
            if Curr_Q2 > Best_Q2:
                Optimal_Components = Components_Num
                Best_Q2 = Curr_Q2
                
                if Verbose == True:
                    print('Optimal Components Update: ' + str(Optimal_Components))
                    print('Optimal Q2 Update: ' + str(Best_Q2))
                    print('\n')
                    pass
                pass
            
            continue
        
        #Instantiate and train final PCA model using all data provided
        super().__init__(n_components=Optimal_Components)
        self.fit(X_Data)
        
        return
    
    
    def score(self, X_Test):
        
        #Acquire boolean arrays for missing data
        Nulls_Bool_np = np.zeros(shape=X_Test.shape)
        Nulls_Bool_np = np.isnan(X_Test)
        NonNulls_Bool_np = 1 - Nulls_Bool_np
        
        #Variable Indices
        Var_Indices = np.linspace(start=0, stop=X_Test.shape[1]-1, num=X_Test.shape[1], dtype='int')
        
        #Residual/Total sum of squares initialization
        RSS = 0
        TSS = 0
        
        #Loop over variable folds
        Var_Folder = KFold(n_splits=self.Folds_Num)
        np.random.shuffle(Var_Indices)
        for Input_Group, Output_Group in Var_Folder.split(Var_Indices):
            
            #KFold.split() only returns indices of Obs_Indices
            #Does not actually split Obs_Indices into train/test arrays
            Input_Indices = Var_Indices[Input_Group]
            
            #Acquire corresponding input/output boolean arrays
            Inputs_Bool_np = np.zeros(shape=X_Test.shape)
            Inputs_Bool_np[:, Input_Indices] = 1
            Outputs_Bool_np = 1 - Inputs_Bool_np
            
            #Initialize input data, effectively setting output columns to zero
            Input_Data = X_Test * Inputs_Bool_np
            
            #Set all missing data values to zero initially
            Input_Data[Nulls_Bool_np] = 0
            
            #Convergence
            for i in range(3):
                Latent_Data = super().transform(Input_Data)
                Pred_Data = super().inverse_transform(Latent_Data)
                
                #Update all output variable data
                Input_Data = Input_Data * Inputs_Bool_np + Pred_Data * Outputs_Bool_np
                
                #Update all missing value data
                Input_Data = Input_Data * NonNulls_Bool_np + Pred_Data * Nulls_Bool_np
                continue
            
            #Set null values to zero to get rid of nan in summation
            X_Test[Nulls_Bool_np] = 0
            
            
            RSS += np.sum(NonNulls_Bool_np * Outputs_Bool_np * (X_Test - Pred_Data)**2)
            TSS += np.sum(NonNulls_Bool_np * Outputs_Bool_np * (X_Test**2))
            
            continue
        
        Q2 = 1 - RSS / TSS
        return Q2
    
    
    def transform(self, X_Data):
        #Can handle missing data
        
        Nulls_Bool_np = np.isnan(X_Data)
        NonNulls_Bool_np = 1 - Nulls_Bool_np
        
        #Initialize missing values to zero
        X_Data[Nulls_Bool_np] = 0
        
        Iterations = 3
        for i in range(Iterations):
            
            X_Latent = super().transform(X_Data)
            X_Pred = super().inverse_transform(X_Latent)
            X_Data = X_Data * NonNulls_Bool_np + X_Pred * Nulls_Bool_np
            continue
        
        return X_Latent

