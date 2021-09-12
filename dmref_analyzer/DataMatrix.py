import pandas as pd
import os
import sys
import dmref_analyzer.util as util
import numpy as np
import math
from sklearn.preprocessing import scale
from os.path import dirname, abspath

default_param_rule = pd.read_csv(
    os.path.expanduser('./parameter_rules.csv'), index_col=1)
experiment_types = default_param_rule.ix['experimentType', 'options'] \
    .split(',')
experiment_types = list(map(lambda x: x.strip(), experiment_types))
# For generating outcomes for RF and R, not officially in parameter_rules
experiment_types += ['genRF', 'genR', 'RF', 'R']

default_features = dict()
default_outcomes = dict()
default_input_dir = dirname(dirname(abspath(__file__)))
default_features['TenS'] = [u'density',
                            u'degreeOfMisalignment',
                            u'averageLength',
                            u'maxLength',
                            u'bundlingExtent']
default_outcomes['TenS'] = [u'linearStrain',
                            u'linearStrengthMPa',
                            u'youngsModulusMPa',
                            u'fractureStrain',
                            u'fractureStrengthMPa']
pressure_ = [u'voltage', u'freq', u'totalNumOfCycles', u'diameter',
             u'lengthTensile', u'pressure']
default_features['TenF'] = pressure_
default_outcomes['TenF'] = [u'linearStrain',
                            u'linearStrengthNtex',
                            u'linearStrengthMPa',
                            u'youngsModulusMPa',
                            u'fractureStrain',
                            u'youngsModulusNtex',
                            u'fractureStrengthMPa',
                            u'fractureStrengthNtex']
default_features['F'] = [u'voltage',
                         u'freq',
                         u'totalNumOfCycles']
default_outcomes['F'] = [u'frequency',
                         u'voltage',
                         u'timeVolOnPerPul',
                         u'dutyCycle']

default_features['R'] = []
default_outcomes['R'] = [u'dGRatio',
                         u'gprimeGRatio',
                         u'dPeakPosition',
                         u'gPeakPosition',
                         u'gprimePeakPosition']

default_features['RF'] = [u'voltage',
                          u'freq',
                          u'totalNumOfCycles']
default_outcomes['RF'] = default_outcomes['R']

default_features['genRF'] = default_features['RF']
default_outcomes['genRF'] = default_outcomes['RF']

default_features['genR'] = []
default_outcomes['genR'] = default_outcomes['R']


class MatrixGenerator(object):
    def __init__(self, data_matrix):
        self.X_train_mean = 0
        self.X_train_std = 0
        self.X_train_generated = False
        self.y_train_mean = 0
        self.y_train_std = 0
        self.y_train_generated = False
        self.data_matrix = data_matrix

    def gen_train(self, features, label, sample_rng):
        filter = self.data_matrix['sampleID'].isin(sample_rng)
        X = self.data_matrix[filter][features].as_matrix()
        y = self.data_matrix[filter][label].as_matrix()
        self.X_train_mean = np.mean(X, axis=0)
        self.X_train_std = np.std(X, axis=0)
        self.X_train_generated = 1
        self.y_train_mean = np.mean(y)
        self.y_train_std = np.std(y)
        self.y_train_generated = 1
        self.features = features
        self.label = label
        return scale(X), scale(y)

    def gen_x_test(self, features, sample_rng):
        filter = self.data_matrix['sampleID'].isin(sample_rng)
        X = self.data_matrix[filter][features].as_matrix()
        if self.X_train_generated:
            return (X - self.X_train_mean) / self.X_train_std
        else:
            raise Exception('Training data needs to be generated first')

    def convert_y_pred(self, y_pred):
        if self.y_train_generated:
            return (y_pred * self.y_train_std) + self.y_train_mean
        else:
            raise Exception('Training data needs to be generated first')


class DataMatrix(object):
    def __init__(self,
                 data_frame=None,
                 experiment_type=None,
                 sample_rng=None,
                 features=None,
                 outcomes=None,
                 input_dir=None,
                 param_rule_file=None):

        self.X_train_generated = False
        self.y_train_generated = False

        if data_frame is None:
            # When no data frame is passed in, experiment_type and sample_rng
            # must not be None
            if experiment_type is None or sample_rng is None:
                raise Exception('Both experiment_type and sample_rng ' +
                                'must be given when no data matrix is passed')

            self.data_matrix = pd.DataFrame()
            self.scaled_data_matrix = pd.DataFrame()
            self.experiment_type = experiment_type
            self.sample_rng = sample_rng

            if self.experiment_type not in experiment_types:
                raise Exception('Experiment type ' + self.experiment_type +
                                ' is not supported')

            if self.experiment_type in ['genRF', 'genR']:
                import matlab.engine
                self.matlab_engine = matlab.engine.start_matlab()
                self.matlab_engine.addpath(
                    dirname(dirname(abspath(__file__))))

            self.features = features if features \
                else default_features[experiment_type]

            self.outcomes = outcomes if outcomes \
                else default_outcomes[experiment_type]

            self.input_dir = input_dir if input_dir \
                else default_input_dir

            self.param_rule = pd.read_csv(param_rule_file, index_col=1) \
                if param_rule_file else default_param_rule

            self.gen_data_matrix()

        elif data_frame is not None:
            # When data_matrix is given, the features and outcomes must all
            # be given
            if features is None or outcomes is None:
                raise Exception('Features, outcomes and sample range ' +
                                'must be given together with the data matrix')
            if 'sampleID' not in data_frame.columns:
                raise Exception('A sampleID column is needed')
            if experiment_type is None:
                raise Exception('Please provide the experiment type')
            self.features = features
            self.outcomes = outcomes
            self.experiment_type = experiment_type
            self.data_matrix = data_frame
            self.sample_rng = sample_rng if sample_rng is not None \
                else data_frame['sampleID'].values

        self.data_matrix_cleaned = \
            self.data_matrix.dropna(axis=1, how='all')
        for col in self.features:
            if np.any(self.data_matrix_cleaned[col].isnull()):
                print('Feature ' + col +
                      ' contains NaN, filled in with average value')
                self.data_matrix_cleaned[col].fillna(
                    self.data_matrix_cleaned[col].mean(), inplace=True)

            # if util.skewness(self.data_matrix_cleaned[col])[0] > 20:
            #     self.data_matrix_cleaned[col] = \
            #         np.log(self.data_matrix_cleaned[col])

            # if col == 'Energy':
            #     self.data_matrix_cleaned[col] = \
            #         np.log(self.data_matrix_cleaned[col])

        self.matrix_generator = MatrixGenerator(self.data_matrix_cleaned)

    def mat(self):
        return self.data_matrix

    def cleaned_mat(self):
        return self.data_matrix_cleaned

    def gen_train(self, features, label, sample_rng=None):
        if sample_rng is None:
            return self.matrix_generator.gen_train(features, label,
                                                   self.sample_rng)
        else:
            return self.matrix_generator.gen_train(features, label, sample_rng)

    def gen_x_test(self, features, sample_rng):
        return self.matrix_generator.gen_x_test(features, sample_rng)

    def convert_y_pred(self, y_pred):
        return self.matrix_generator.convert_y_pred(y_pred)

    def scaled_mat(self, features=None, outcomes=None):
        features = features if features else self.features
        outcomes = outcomes if outcomes else self.outcomes
        self.scaled_data_matrix = \
            util.gen_normalized_dataset(self.data_matrix, features, outcomes)
        return self.scaled_data_matrix

    def gen_data_matrix(self):
        df_list = []
        for sample_id in self.sample_rng:
            param_file_pattern = "Parameters_" + str(
                sample_id) + "(_\d)*" + ".csv"
            param_file = util.find_file_with_regex(self.input_dir,
                                                   param_file_pattern,
                                                   sample_id)
            if not param_file:
                print("param_file for sampleID: " +
                      str(sample_id) + " does not exist")
                continue

            # when genR and genRF, I still need to read the data files of
            # of R and RF
            if self.experiment_type == 'genR':
                data_file_pattern = 'R_' + str(sample_id) + "(_\d)*" + ".csv"
            elif self.experiment_type == 'genRF':
                data_file_pattern = 'RF_' + str(sample_id) + "(_\d)*" + ".csv"
            else:
                data_file_pattern = self.experiment_type + '_' + \
                    str(sample_id) + "(_\d)*" + ".csv"

            data_file = util.find_file_with_regex(self.input_dir,
                                                  data_file_pattern,
                                                  sample_id)
            if data_file is None:
                raise Exception(
                    "No files found with pattern: " + data_file_pattern)

            df = self.gen_data_from_file(param_file, data_file)

            if df is not None:
                df_list.append(df)

        if len(df_list) != 0:
            dm = pd.concat(df_list, ignore_index=True). \
                apply(lambda x: pd.to_numeric(x, errors='ignore'))
            # dm = dm[(dm.length != 15) & (dm.totalNumOfCycles != 15200)]
            self.data_matrix = dm
        # when genR and genRF is called, we do not expect any df being returned
        elif 'gen' in self.experiment_type:
            print(self.experiment_type + ' is done')
            sys.exit(0)
        else:
            raise Exception('Experiment type ' + self.experiment_type +
                            ' does not exist in the specified samples range')

    def gen_data_from_file(self, param_file, data_file=None):
        if self.experiment_type in ['Ten', 'TenF', 'TenS']:
            return self.gen_tensile_data_from_file(param_file, data_file)
        elif self.experiment_type == 'F':
            return self.gen_fusion_data_from_file(param_file, data_file)
        elif self.experiment_type in ['R', 'genR', 'RF', 'genRF']:
            return self.gen_raman_data_from_file(param_file, data_file)
        else:
            raise Exception('Experiment type ' + self.experiment_type +
                            ' is not supported')

    def gen_raman_data_from_file(self, param_file, data_files):
        base_dir = os.path.dirname(param_file)
        sample_idx = int(os.path.dirname(param_file).split('/')[-1])
        outcomes_path = os.path.join(base_dir,
                                     'Outcomes_' + str(sample_idx) + '.csv')
        if 'gen' in self.experiment_type:
            param = pd.read_csv(param_file)
            # actual experiment type includes R1 to R5 or RF1 to RF5
            actual_exp_type = self.experiment_type[3:]
            r_list = [actual_exp_type + str(x) for x in np.arange(1, 6)]
            raman_param = param[param['experimentType'].
                                astype(str).isin(r_list)].reset_index(drop=True)
            if raman_param.empty:
                return None
            for file_path in data_files:
                file_name = os.path.basename(file_path)
                i = int(file_name.split('_')[-1].split('.')[0]) - 1
                raman_outcomes = self.matlab_engine.genRamanData(file_path)[0]
                for j, col_name in enumerate(self.outcomes):
                    raman_param.loc[
                        raman_param['experimentType'] == r_list[i],
                        col_name] = raman_outcomes[j]
            raman_param.to_csv(outcomes_path, index=False)
            return None
        else:
            return pd.read_csv(outcomes_path)

    def gen_fusion_data_from_file(self, param_file, data_file=None):
        param = pd.read_csv(param_file)
        fusion_param = param[param['experimentType'] == self.experiment_type]
        if fusion_param.empty:
            return
        fusion_param_series = fusion_param.iloc[0, :]
        sample_idx = int(fusion_param_series.sampleID)
        # Generate base_df whose data needed to be calculated from the data file
        base_df = pd.DataFrame(columns=['sampleID',
                                        # 'freq',
                                        # 'voltage',
                                        'timeVolOnPerPul',
                                        'dutyCycle'])
        base_df.loc[0, 'sampleID'] = sample_idx

        # Merge the calculated data with the base_df
        df = pd.merge(base_df, fusion_param)

        if data_file:
            fusion_data = pd.read_csv(data_file)
            # print fusion_data
            if 'Unnamed: 4' in fusion_data.columns:
                del fusion_data['Unnamed: 4']
            if 'Item' in fusion_data.columns:
                del fusion_data['Item']
            fusion_data.columns = ['time', 'voltage', 'i']
            times = fusion_data['time']
            voltage = fusion_data['voltage'].max()
            loop_len = 4
            freq = 1 / np.mean(np.diff(times.iloc[::loop_len]))
            even = np.arange(2, len(times) - 1, 2)
            odd = np.arange(1, len(times) - 1, 2)
            time_voltage_on_per_pulse = \
                np.mean(times[even].as_matrix() - times[odd].as_matrix())
            total_time_voltage_on = \
                np.sum(times[even].as_matrix() - times[odd].as_matrix())
            total_time = times.iloc[-1] - times.iloc[0]
            duty_cycle = total_time_voltage_on / total_time
            df.loc[0, 'freq'] = freq
            df.loc[0, 'voltage'] = voltage
            df.loc[0, 'timeVolOnPerPul'] = time_voltage_on_per_pulse
            df.loc[0, 'dutyCycle'] = duty_cycle
        return df

    def pre_process_ten(self, data_file, ten_param_series):
        if self.experiment_type == 'TenF' or self.experiment_type == 'Ten':
            # the two columns in the TenF.csv or Ten.csv files are:
            # displacement and force
            ten_data = pd.read_csv(data_file, names=['displacement', 'force'])
            linear_density = ten_param_series.linearMassDensity
            length_tensile = ten_param_series.lengthTensile
            # diameter from the ten_param_series is in micrometer diameter in mm
            diameter = ten_param_series.diameter / 1e3
            # width is in mm
            width = ten_param_series.width
            # thickness in mm
            thickness = ten_param_series.thickness
            # If there is no width, it is a yarn,
            # so we are calculating the cross section as a circle
            if np.isnan(width):
                area = .25 * math.pi * pow(diameter, 2)
            # Otherwise it is a roving,
            # we are calculating the cross section as a rectangle
            else:
                area = width * thickness

            force = ten_data.force.as_matrix()
            starting_point = np.nonzero(force > .05)[0][0]
            # starting_point = force[ten_data.force > .05].index[0]
            ending_point = force.argmax()
            # this is an array of strain that is generated
            # by subtracting the first displacement
            # and divide that by the length, it is unitless
            displacement = ten_data.displacement. \
                as_matrix()[starting_point: ending_point + 1]
            strain = (displacement - displacement[0]) / length_tensile
            # this is the array of the force
            # force is in Newton
            force = force[starting_point: ending_point + 1]
            # stress is generated by dividing force by cross-section area
            # stress is in MPa
            stress = force / area
            # up to this point, we are doing data cleaning / trimming,
            # which basically cut the data and convert it into strain and stress
            return linear_density, area, stress, strain
        elif self.experiment_type == 'TenS':
            # The two columns in TenS.csv are instead strain and stress
            ten_data = pd.read_csv(data_file, names=['strain', 'stress'])
            # Polynomial fit the data with degree == 6
            p = np.polyfit(ten_data['strain'], ten_data['stress'], 6)
            strain = ten_data['strain'].as_matrix()[0:len(ten_data):5]
            stress = np.polyval(p, strain)
            # ending_point is the peak AFTER POLYNOMIAL FIT
            ending_point = stress.argmax()
            strain = strain[0: ending_point + 1]
            stress = stress[0: ending_point + 1]
            # stress = ten_data['stress'].as_matrix()[0: ending_point + 1]
            # strain = ten_data['strain'].as_matrix()[0: ending_point + 1]
            return [stress, strain]

    def gen_tensile_data_from_file(self, param_file, data_file=None):
        param = pd.read_csv(param_file)
        ten_param = param[param['experimentType'] == self.experiment_type]
        if ten_param.empty:
            return
        ten_param_series = ten_param.iloc[0, :]
        sample_idx = int(ten_param_series.sampleID)
        base_df = pd.DataFrame(
            columns=['sampleID', 'linearStrain', 'linearStrengthMPa',
                     'linearStrengthNtex', 'youngsModulusMPa',
                     'youngsModulusNtex',
                     'fractureStrain', 'fractureStrengthMPa',
                     'fractureStrengthNtex',
                     'yieldStiffnessMPa', 'yieldStiffnessNtex'])
        base_df.loc[0, 'sampleID'] = sample_idx
        df = pd.merge(base_df, ten_param)

        if data_file:
            if self.experiment_type == 'Ten' or self.experiment_type == 'TenF':
                [linear_density, area, stress, strain] = \
                    self.pre_process_ten(data_file, ten_param_series)
            elif self.experiment_type == 'TenS':
                [stress, strain] = \
                    self.pre_process_ten(data_file, ten_param_series)

            # What Sanwei did: make stress and strain starts at 0
            stress = stress - stress[0]
            strain = strain - strain[0]

            # print "Pre-Processing time: ", time.time() - start
            # print "Sample ID: ", sample_idx

            # now we are trying to find the young's modulus
            # (the elbow point that divides the first slope and the second slope)
            strain_interval = 50
            # interval_start = int(data_length / max(strain) / 100)
            interval_start = 0
            data_length = len(strain)
            stiffness_candidates = np.zeros(data_length)

            # stiffness_determine is basically an array of slopes value for
            # 0-inteveral, 1-interval+1, 2-interval+2, ...
            for data_idx in np.arange(interval_start,
                                      data_length - strain_interval):
                stiffness_candidates[data_idx + strain_interval] = \
                    (stress[data_idx + strain_interval] - stress[data_idx]) / \
                    (strain[data_idx + strain_interval] - strain[data_idx])
            # for data_idx in range(interval_start, data_length - strain_interval):
            #     stiffness_determine = np.append(stiffness_determine, (stress[data_idx + strain_interval] - stress[data_idx]) /
            #         (strain[data_idx + strain_interval] - strain[data_idx]))

            stiffness_value = max(stiffness_candidates)
            stiffness_idx = stiffness_candidates.argmax()

            linear_youngs_modulus = stiffness_value
            # Find the line that goes through the point that is .002 to the right of the largest stiffness point
            # The slope of this line is linear young's modulus

            line_for_linear_strain = linear_youngs_modulus * \
                (strain - strain[stiffness_idx] - .002) + \
                stress[stiffness_idx]

            # Find the intersection point of the offset line to the original stress/strain line
            intersect_point_idx = \
                np.nonzero((line_for_linear_strain - stress) >= 0)[0][0]

            linear_strain = strain[intersect_point_idx]
            linear_strength = stress[intersect_point_idx]
            fracture_strain = max(strain)
            fracture_strength = max(stress)

            # Need to update the way to calculate the yield_stiffness in the future
            yield_stiffness = (max(stress) - linear_strength) / (
                max(strain) - linear_strain)

            df.loc[0, 'linearStrain'] = linear_strain
            df.loc[0, 'linearStrengthMPa'] = linear_strength
            df.loc[0, 'youngsModulusMPa'] = linear_youngs_modulus
            df.loc[0, 'fractureStrain'] = fracture_strain
            df.loc[0, 'fractureStrengthMPa'] = fracture_strength
            df.loc[0, 'yieldStiffnessMPa'] = yield_stiffness

            if self.experiment_type == 'Ten' or self.experiment_type == 'TenF':
                df.loc[0, 'linearStrengthNtex'] = \
                    linear_strength * area / linear_density
                df.loc[0, 'youngsModulusNtex'] = \
                    linear_youngs_modulus * area / linear_density
                df.loc[0, 'fractureStrengthNtex'] = \
                    fracture_strength * area / linear_density
                df.loc[0, 'yieldStiffnessNtex'] = \
                    yield_stiffness * area / linear_density
        return df
