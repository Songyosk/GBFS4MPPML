"""
Module to query and retrieve chemical mateirals and their properties from Materials Project.

More information can be found here: https://docs.materialsproject.org/open-apis/the-materials-api/#mpquery

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os
import requests
import pandas as pd
import json  

from pymatgen.ext.matproj import MPRester
from lxml import html


class query_MP():
    """
    Class to query and retrieve chemical materials and their properties from MaterialsProject

    args:
        (1) API_key from MaterialsProject (type:str) - API key to access the MaterialsProject database
        (2) general_formula e.g. 'AB2' (type:str) - directory name; keep it consistent with 'anonymous_formula' e.g. AB2
        (3*) anonymous_formula e.g. "{'A': 1.0, 'B': 2.0}" (mongodb query) - general chemical formula to search
        (4*) elements e.g. ['Fe', 'O'] (list) - list of elements that are to be found in the composition
        
    return: 
        (1) materials properties (csv & json)
        (2) CIFs
        (3) magnetic order type (csv)


    (i) Example based on anonymous_formula:
    data = Query_MP('qfqsDIFq4Fef4Zvm26DQz', 'AB2', "{'A': 1.0, 'B':2.0}")
    data.Retrieve_Properties()
    data.Get_CIF()

    (ii) Example based on elements:
    data = Query_MP('qfqsDIFq4Fef4Zvm26DQz', 'AB2', elements=['Fe', 'O'])
    data.Retrieve_Properties()
    data.Get_CIF()

    (iii) Example based on both anonymous_formula and elements:
    data = Query_MP('qfqsDIFq4Fef4Zvm26DQz', 'AB2', anonymous_formula="{'A':1.0, 'B':2.0}", elements=['Fe', 'O'])
    data.Retrieve_Properties()
    data.Get_CIF()
    """

    def __init__(self, API_key, general_formula, *args, **kwargs):
        self.API_key = API_key #e.g. qfqsDIFq4Fef4Zvm26DQ
        self.general_formula = general_formula
        self.directory = os.path.join('retrieved_data', self.general_formula)
        self.anonymous_formula = kwargs.get('anonymous_formula') 
        self.elements = kwargs.get('elements')


        #Properties of interest
        fields = [
            'task_id', 
            'pretty_formula',  
            'cif', 
            'total_magnetization',
            'formation_energy_per_atom',
            'e_above_hull',
            'energy_per_atom',
            'nsites',
            'nelements',
            'volume',
            'density',
            'spacegroup.number',
            'band_gap',
            'elasticity.G_Voigt_Reuss_Hill',
            'elasticity.K_Voigt_Reuss_Hill',
            'elasticity.poisson_ratio', 
            'diel.poly_electronic', 
            'diel.n'
            ]


        #Query MP
        criterion_1 = "'anonymous_formula': {'$in': [" 
        criterion_1 += self.anonymous_formula if self.anonymous_formula else '' 
        criterion_1 += ']}'


        if self.anonymous_formula == None:
            criterion_1 = '' 
            criterion_2 = " 'elements': {'$all': ["

        elif self.anonymous_formula != None:
            criterion_2 = ", 'elements': {'$all': ["


        if self.elements == None:
            criterion_2 = '' 

        elif self.elements != None:
            i = 0 
            while i < len(self.elements):
                criterion_2 += "'" + str(self.elements[i]) + "'"

                if i < len(self.elements) - 1:
                    criterion_2 += ','
                i = i + 1

            criterion_2 += ']}'      


        query = eval("{" + criterion_1 + criterion_2 + "}")

        print('Made the followinng query: ', query)


        with MPRester(self.API_key) as mp:
            self.retrieved_data = mp.query(criteria = query, properties = fields)

        print('Total number of materials: ', len(self.retrieved_data))



    def retrieve_properties(self):
        """
        Retrieve material properties & saves as csv & json
        """

        #Create directory
        directory = os.path.join('retrieved_data', self.general_formula)
        os.makedirs(directory, exist_ok = True)


        #Save data as csv 
        self.property_names = []

        for i in self.retrieved_data[0].keys():
            if i != 'cif':
                self.property_names.append(i)


        self.csv = self.general_formula + '.csv'


        with open(os.path.join(directory, self.csv), 'w+') as wf:
            wf.write(','.join(self.property_names))
            wf.write('\n')


        for i in self.retrieved_data:
            with open(os.path.join(directory, self.csv), 'a+') as wf:
                values = [str(i[name]) for name in self.property_names]
                wf.write(','.join(values))
                wf.write('\n')


        #Save data as json
        with open(os.path.join(directory, self.general_formula + '.json'), 'w+') as wf:
            json.dump(self.retrieved_data, wf)



    def get_cif(self):
        """
        Write Chemical Information Files (CIFs) of the corresponding materials
        """

        #Create directory
        directory = os.path.join('retrieved_data', self.general_formula)
        os.makedirs(os.path.join(directory, 'cifs'), exist_ok = True)


        #Save CIFs
        for i in self.retrieved_data:
            with open(os.path.join(directory, 'cifs', i['task_id'] + '.cif'), 'w+') as wf:
                wf.write(i['cif'])



    def get_magnetic_order(self):
        """
        Web scrapping of MaterialsProject to retrieve the magnetic order of the materials & save as csv
        Note that xpath can change in the future and may need to be updated
        """

        cif_id = [i['task_id'] for i in self.retrieved_data]
        magnetic_order = []
        material_id = []
        error_id = []


        for i in cif_id:
            try:
                page = requests.get('https://materialsproject.org/materials/' + i + '/')

                tree = html.fromstring(page.content)
                mag_order = tree.xpath('/html/body/div/div/div[3]/div[1]/div[2]/div[2]/table[1]/tbody/tr[2]/td/span/text()')
                result = mag_order[0].replace(' ', '').replace('\n', '')
                
                magnetic_order.append(result)
                material_id.append(i)

                print('Magnetic order of ', i, ' is ', result)

            except:
                error_id.append(i)

                print('Error with ', i)


        #Merge data
        df1 = pd.DataFrame(cif_id, columns = ["task_id"]).set_index('task_id', drop = True)
        df2 = pd.DataFrame({
            'task_id': material_id,
            'mag_order': magnetic_order
            })


        df_mag = pd.merge(df1, df2, left_index = True, right_on = 'task_id', how = 'outer')
        df_mag.reset_index()


        #Remove empty rows
        df_mag = df_mag[df_mag.mag_order != 'Unknown']
        df_mag = df_mag[df_mag.mag_order != None]


        #Save data as csv
        df_mag.to_csv(os.path.join(self.directory, r'magnetic_order_' +  str(self.general_formula) + '.csv'), index = False)

