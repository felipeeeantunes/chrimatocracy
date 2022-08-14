 # Introduction

 Recent advances on technologies behind the Internet and mobile phone allowed bilions of people to conect and interact to each other in way never seen before. As a consequence, entirely new scales of data are generated daily, for example Facebook alone reports 1.32 billion daily active users on average for June 2017 \footnote{https://newsroom.fb.com/company-info/} and in {self.year} reported about 600 TB daily rate of data \footnote{https://code.facebook.com/posts/229861827208629/scaling-the-facebook-data-warehouse-to-300-pb/}. The need for methodologies in order to extrat information (knowledge) from such data's volume and variety scaled at the same rate and scientific fields, like physics, whose experience of huge amounts of data generated by experiments like CERN that generate ~ 25GB of data per second or ~2 PB daily \footnote{https://home.cern/about/computing/processing-what-record} or Dark Energy Survey that report 2PB of data to analise \footnote{http://lss.fnal.gov/archive/2012/pub/fermilab-pub-12-930-a.pdf}, have a lot of methods at disposal. 
 
 
 Recently, several Brazil's representatives are facing accusations of receive campaign donations 
 in exchange of favors and/or to laundry money. 
 
 A corruption scandal involving millions of dollars 
 in kickbacks and more than 80 politicians and members of the business elite. Odebrecht, 
 the Brazilian-based construction giant, which is Latin America's largest construction conglomerate, 
 has admitted bribing officials to secure contracts in Brazil and other countries in South America. 
 Marcelo Odebrecht, former CEO, says part of the $48m he donated to both Ms Rousseff's and Mr Temer's 
 campaigns in the {self.year} Brazilian presidential election was illegal.
 
 # CONETION BETWEEN DATA SCIENCE, STATISTICAL PHYSICS AND POLITICS HERE
 
 
 Since campaign donations play a vital rule for an election of a given candidate\cite{gammerman-antunes} 
 and can be made directly by persons or companies or indirectly by intermediary institutions, 
 such as electoral committees and other companies linked to political parties, Complex Network Theory can
 serve as a tool to analyze how the amounts received are distributed among the candidates. 
 
 
 
 The Complex Network Theory has driven both qualitative and quantitative analysis of party-political networks, 
 encompassing investigations ranging from a description of politicians behavior in polls to the detection of communities that 
 arise from the pattern of bills co-sponsorship 
 \cite{DalMaso{self.year}, Chessa{self.year}, Bursztyn2015a, Zhang{self.min_obs}8a, Waugh{self.min_obs}9, Porter{self.min_obs}5c, Porter{self.min_obs}7a, Fowler{self.min_obs}6, Waugh{self.min_obs}9}.
 
 In particular, community detection is one of the main tasks of Complex Network Theory and 
 such methods don't need incorporate any specific knowledge about committee members or 
 political parties, providing a suitable approach for a unbiased scientific investigation. 

 # Data

 FAZER ISSO PARA CADA COMUNIDADE
 
 Inserir histograma de percentual de empreasas vs numero de candidatos favorecidos
 Fig. 1 shows histograms of the total number of bills sponsored by each legislator in each
 Congress for the House and Senate on a logarithmic plot. For comparability between the House
 and Senate the counts are converted to percent of chamber and pooled across Congresses, but
 the distributions for individual Congresses (not shown) tell the same story. These distributions
 are clearly not power law distributed. In contrast to the large number of scholars who publish
 five scientific papers or less, most legislators sponsor five bills or more (91% of legislators in the
 House and 99% of legislators in the Senate).
 
 ----
 Numero de doadores por candidato
 Fig. 2 shows the distribution of the number of cosponsors per bill on a log–log plot. To aid
 in comparing the House and Senate, the number of cosponsors is divided by the total number of
 legislators in the chamber. Notice that the distributions for the House and Senate are quite close,
 suggesting that the cosponsorship process in both branches is similar. In fact, for bills cosponsored
 by up to 49% of the chamber, these distributions look like the power law distributions of number
 of coauthors per article found in the scientific authorship literature. A simple log–log regression
 of cosponsors as a percent of chamber on frequency of bills suggests a power law exponent of
 γ = 1.69 (S.E. 0.03, R2 = 0.94) in the House and γ = 1.59 (S.E. 0.04, R2 = 0.97) in the Senate

 The network analyzed in this paper were assembled using data made available on-line by the TSE. For each federal election (from {self.min_obs}2 to 2016), the TSE provides two datasets for each federal unity and the country, containing electoral campaign spending and recipes. In this work we focus in donations made during electoral campaign of {self.year}, summarized above: 

 From the data, we can have donations made in two possible ways:
 
 
 - Direct, where a company (characterized by a CNPJ number of 14 digits) or a person (characterized by a CPF of 11 digits) donate to a candidate;
 - Indirect, where a donation made by a company or a person are made to a committee or party representative legal institution (characterized by a CNPJ number of 14 digits) then reach the candidate.
 
 
 The fields of interest are \it{CPF candidato}, \it{CPF/CNPJ do doador} and \it{CPF/CNPJ do doador orinário}, the first representing the candidate, the second representing the intermediary legal entity when it exists or the donator if the donation are made directly, and the third representing the original donator when the intermediary are present. 

 {'CNPJ Prestador Conta': 'id_accountant_cnpj',
  'CPF do candidato': 'id_candidate_cpf',
  'CPF/CNPJ do doador': 'id_donator_cpf_cnpj',
  'CPF/CNPJ do doador originário': 'id_original_donator_cpf_cnpj',
  'Cargo': 'cat_political_office',
  'Cod setor econômico do doador': 'id_donator_economic_sector',
  'Cód. Eleição': 'id_election',
  'Data da receita': 'dat_donation_date',
  'Data e hora': 'dat_donation_date_time',
  'Desc. Eleição': 'cat_election_description',
  'Descricao da receita': 'cat_donation_description',
  'Especie recurso': 'cat_donation_method',
  'Fonte recurso': 'cat_donation_source',
  'Nome candidato': 'cat_candidate_name',
  'Nome do doador': 'cat_donator_name',
  'Nome do doador (Receita Federal)': 'cat_donator_name2',
  'Nome do doador originário': 'cat_original_donator_name',
  'Nome do doador originário (Receita Federal)': 'cat_original_donator_name2',
  'Numero Recibo Eleitoral': 'id_receipt_num',
  'Numero candidato': 'id_candidate_num',
  'Numero do documento': 'id_document_num',
  'Número candidato doador': 'id_donator_number',
  'Número partido doador': 'id_donator_party',
  'Sequencial Candidato': 'id_candidate_seq',
  'Setor econômico do doador': 'cat_donator_economic_sector',
  'Setor econômico do doador originário': 'cat_original_donator_economic_sector',
  'Sigla  Partido': 'cat_party',
  'Sigla UE doador': 'cat_donator_state',
  'Tipo doador originário': 'cat_original_donator_type',
  'Tipo receita': 'cat_donation_type',
  'UF': 'cat_federative_unity',
  'num_donation_ammount': 'num_donation_ammount',
  'doador': 'id_donator_effective_cpf_cnpj'}

# Parties donations' Network 
         
    Adapting the approaches of \cite{Bursztyn2015a} and \cite{Zhang{self.min_obs}8a}, the data was employed to create two networks: 
    
    
    - A (bipartite) directed one where a legal entity is connected by an edge to each candidate it sponsored or cosponsored. This is encoded using a bipartite adjacency matrix $M$, with entries $M_{ij}$ equal to $1$ if legal entity $j$ donated to candidate $i$ and $0$ otherwise. 
    
    - A second (unipartite), undirected one, projected from the first, with adjacency matrices $A_{ij} = \sum_k M_{ik} M^T_{kj}$ in which nodes are candidates and the weighted edges connecting them indicate how many times they received money together from the same legal entity $k$.ipynb_checkpoints/
     
    The donnation network forms a bipartite network, a donor is connected by an edge to candidate it sponsored or cosponsored. This is encoded using a bipartite adjacency matrix M, with entries Mij equal to 1 if legal entity i (co-)sponsored candidate j and 0 if not.
    We can analyze the cosponsorship networks using one-mode (“unipartite”) projections with adjacency matrices Aij, in which nodes are candidates and the weighted edges connecting them indicate how many times they shared donations from the same donors.
     
    We generated the transactions network using the Python's library NetworkX\cite{NetworkX} and drew it using the interactive visualization and exploration platform Gephi\cite{Gephi}. 
    To draw the graph we employed force-directed placement methods. We began giving more space to the graph and avoiding a randomized start using Fruchterman Reingold's algorithm \cite{Fruchterman}. After the system reached mechanical equilibrium, we applied Force Atlas 2 \cite{ForceAtlas2} to produce an aesthetically pleasant and intuitive graph.
    To get a visual information about the importance of each node, its sizes was chosen to reflect the number of votes the corresponding candidate received. Finally, in order to detect patterns in donations we chose colors corresponding to communities extracted employing the Louvain Method for community detection \cite{Louvian}. 