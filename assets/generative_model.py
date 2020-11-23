import numpy as np
import csv
import pandas as pd
import benford as bf
import matplotlib.pyplot as plt


class BenfordGenerativeModel:

    def __init__(self, role, data_path, table_path, figure_path, group):

        self.role = role
        self.group = group
        self.data_path = data_path
        self.figure_path = figure_path
        self.table_path = table_path

    @staticmethod
    def Fun(X, gamma, eta0):
        delta  = np.log(0.01)
        etaMax = np.log(max(X.values))

        A= 1+(eta0/(etaMax-delta))**gamma
        B= 1+(eta0/(np.log(X.values)-delta))**gamma
        
        f = A/B
        return f

    
    @staticmethod
    def drand_distr(gamma, csi0, csim=100., delt=0.):
        # This generates a random number with distribution according to eq 6 in the manuscript
        F = np.random.uniform(0,1)
        bla = csi0/(csim-delt)
        blabla = F/(bla**gamma+1-F)
        
        return np.exp(csi0*blabla**(1./gamma)+delt)

    @staticmethod
    def func(x, gamma, csi0, csim=100., delt=0.):
        # this is F(x) in eq. 5
        # cumulative distribution
        num1 = np.log(x)-delt
        num2 = (csi0/num1)**gamma
        num3 = (csi0/(csim-delt))**gamma
        term2 = 1.+num3
        term3 = 1./(num2+1.)
        return term2*term3

    
    @staticmethod
    def func2(x, gamma, csi0, csim=100., delt=0.):
        # this is f(x) in eq. 6
        # distribution
        num1 = np.log(x)-delt
        num2 = (csi0/num1)**gamma
        num3 = (csi0/(csim-delt))**gamma
        term1 = gamma/x
        term2 = 1.+num3
        term3 = num2/( (num2+1.)**2 )
        term3 /= num1
        return term1*term2*term3

        
    @staticmethod   
    def lkhd_distr(nums, gamma, csi0, csim, delt):
        # This evaluates lnL and its gradient
        sum1 = 0.
        sum2 = 0.
        sum3 = 0.
        sum4 = 0.
        sum5 = 0.
        sum6 = 0.
        N = len(nums)
        num1 = csi0/(csim-delt)
        num1g = num1**gamma
        termb = num1g/(1.+num1g)
        for ele in nums:
            num2 = np.log(ele)
            num3 = num2-delt
            num4 = csi0/(num3)
            num4g = num4**gamma
            term1 = num4g/(num4g+1.)
            # for lkhd
            sum1 += num2
            sum2 += np.log(num3)
            sum3 += np.log(1.+num4g)
            # for dlnldcsi0
            sum4 += term1
            # for dlnldgamma
            sum5 += np.log(num4)*term1
        lkhd = N*(np.log(gamma)+np.log(1.+num1g)+gamma*np.log(csi0))-sum1-(gamma+1.)*sum2-2.*sum3
        g1 = N*(gamma*termb/csi0 + gamma/csi0)-2.*gamma*sum4/csi0 #csi0
        g2 = -N*gamma*termb/(csim-delt) #csim
        g3 = N*(1./gamma + termb*np.log(num1)+np.log(csi0)) - sum2-2.*sum5 #gamma
        return lkhd, (g1, g2, g3)


    def maxLKHD_distr(self, nums, gamma=3., csi0=3., csim=100., delt=0.01, lamb=1., eps=1.e-4):
        ### This algorithm may be unstable (never finishes) for some small sets of numbers For the ones in the manuscript it is working fine
        # This evaluates parameters for obtaining maximum value of lnL
        lkhd, grad = self.lkhd_distr(nums, gamma, csi0, csim, delt)
        norm = (grad[0]**2+grad[1]**2+grad[2]**2)**.5
        ncsi0 = csi0+lamb*grad[0]/norm
        ncsim = csim#+lamb*grad[1]/norm
        ngamma = gamma+lamb*grad[2]/norm
        nlkhd, ngrad = self.lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
        while lamb*norm > eps:
            #print lamb, norm
            if nlkhd>lkhd: # accepted
                grad = ngrad
                csi0 = ncsi0
                csim = ncsim
                gamma = ngamma
                lkhd = nlkhd
                lamb *= 1.2
            else:
                lamb *= .01
            norm = (grad[0]**2+grad[1]**2+grad[2]**2)**.5
            ncsi0 = csi0+lamb*grad[0]/norm
            ncsim = csim#+lamb*grad[1]/norm
            ngamma = gamma+lamb*grad[2]/norm
            nlkhd, ngrad = self.lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
        return gamma, csi0, csim
    
    def fit(self, data, role):
        self.role = role
        self.df = data[[self.group, 'id_candidate_cpf','num_donation_ammount']].copy()      
        self.df.to_csv(self.data_path / f'{role}_{self.group}_generative_model_input.csv', index=False)
        self.group_list = data[self.group].unique()
        self.df_gen = self.df.copy()

        with open(self.data_path / f'{role}_{self.group}_generative_model_parameters.csv','w') as f1:
            writer=csv.writer(f1, delimiter=',',lineterminator='\n') 
            header = ['xmin', 'xmax', 'gamma', 'eta0']
            writer.writerow(header)
            
            for g in self.group_list:
                cpfs = list(self.df_gen.loc[self.df_gen[self.group] == g, 'id_candidate_cpf'].unique())
                X    = self.df_gen.loc[self.df_gen[self.group] == g, 'num_donation_ammount']
                xmin = min(X.values)
                xmax = max(X.values)
                result = self.maxLKHD_distr(X.values, gamma=1., csi0=1, csim=np.log(xmax), delt=np.log(0.005))
                gamma = result[0]
                eta0 = result[1]
                row = [xmin, xmax, gamma, eta0]
                writer.writerow(row)
                print('Grupo: ',g)
                print('---------------------')
                print('xmin:' ,xmin, 'xmax', xmax)
                print('Gamma:' ,gamma, 'Eta_0', eta0)
                print('---------------------')
                for cpf in cpfs :
                    idx = self.df_gen.loc[(self.df_gen[self.group] == g) 
                                        & (self.df_gen['id_candidate_cpf'] == cpf), 'num_donation_ammount'].index
                    for _idx in idx:
                        self.df_gen.loc[_idx, 'num_donation_ammount'] = self.drand_distr(gamma, eta0, csim=np.log(xmax), delt=np.log(0.005))
            
            print(f"Model parameters saved in {role}_{self.group}_generative_model_parameters.csv")
            self.df_gen.to_csv(self.data_path / f'{role}_{self.group}_generative_model_output.csv', index=False)
        
        return self.df_gen

    def write_latex_tables(self):
        df     = pd.read_csv(self.data_path / f'{self.role}_{self.group}_generative_model_input.csv')
        df_gen = pd.read_csv(self.data_path / f'{self.role}_{self.group}_generative_model_output.csv') 
         
        parameters = pd.read_csv(self.data_path / f'{self.role}_{self.group}_generative_model_parameters.csv')
    
        print('Model Parameters', parameters)
        with open(self.table_path / f'{self.role}_{self.group}_generative_model_parameters.tex', 'w') as tf:
            tf.write(parameters.reset_index().rename(columns={'index':'Group'}).to_latex(index=False))
   
        benford_gen = bf.benford_digits_table(df_gen, self.group)
    
        print('Benford Law Results expected by the model', benford_gen)
        with open(self.table_path / f'{self.role}_{self.group}_generative_model_expected_parameters.tex', 'w') as tf:
            tf.write(benford_gen.reset_index().rename(columns={'index':'Group'}).to_latex(index=False))

    def save_figures(self, show=False):
        
        df     = pd.read_csv(self.data_path / f'{self.role}_{self.group}_generative_model_input.csv')
        df_gen = pd.read_csv(self.data_path / f'{self.role}_{self.group}_generative_model_output.csv') 
        group_list = df[self.group].unique()
        
        fig, axes = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True, figsize=(18,32))
        axes_list = [item for sublist in axes for item in sublist] 

        for g in group_list:
            ax = axes_list.pop(0)
            H2,X2 = np.histogram( np.log(df.loc[df[self.group] == g,'num_donation_ammount']), density=True )
            dx2 = X2[1] - X2[0]
            F2 = np.cumsum(H2)*dx2
            ax.plot(X2[1:], F2, label='data')
            ax.fill_between(X2[1:], F2, alpha=0.2)
            
            H2,X2 = np.histogram( np.log(df_gen.loc[df_gen[self.group] == g,'num_donation_ammount']), density=True )
            dx2 = X2[1] - X2[0]
            F2 = np.cumsum(H2)*dx2
            ax.plot(X2[1:], F2, color = 'r', label='model')
        
            ax.legend()
            ax.set_title('Group Index: '+ str(g))
            ax.set_xlabel("")
            ax.tick_params(
                which='both',
                bottom=False,
                left=False,
                right=False,
                top=False
            )
            ax.grid(linewidth=0.25)
            ax.set_xlabel('$\ln(x)$')
            ax.set_ylabel('$F(x)$')
            ax.set_xlim((-2, 12))
            ax.set_xticks((-2,0,2, 4, 6, 8,10, 12))
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)



        for ax in axes_list:
            ax.remove()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        plt.savefig(self.figure_path / f'{self.role}_{self.group}_generative_model_cdf.pgf')
        print(f"Figure '{self.role}_{self.group}_generative_model_cdf.pgf' saved.")

        if show==True:
            plt.show()