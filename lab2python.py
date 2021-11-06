# Authors: Piotr Michałek s19333 & Kibort Jan s19916

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

'''
Problem: dowiedz się ile wody powinienen wypić człowiek bazując na wieku, wadze osoby oraz temperatury otoczenia.

Wykonaj następujące polecenia:
pip install numpy
pip install -U scikit-fuzzy
pip install -U matplotlib
'''

# Temperature from 0 to 50 celcius degrees
temperature = ctrl.Antecedent(np.arange(0, 50, 1), 'temperature')
temperature['low'] = fuzz.trimf(temperature.universe, [0, 5, 15])
temperature['optimal'] = fuzz.trimf(temperature.universe, [5, 15, 25])
temperature['high'] = fuzz.trimf(temperature.universe, [15, 25, 50])

# Age from 20 to 122 (so far the oldest man ever)
age = ctrl.Antecedent(np.arange(20, 122, 1), 'age')
age['youth'] = fuzz.trimf(age.universe, [20, 25, 30])
age['adult'] = fuzz.trimf(age.universe, [25, 45, 65])
age['old'] = fuzz.trimf(age.universe, [45, 65, 122])

# Weight from 50 to 200 kg
weight = ctrl.Antecedent(np.arange(50, 200, 1), 'weight')
weight['light'] = fuzz.trimf(weight.universe, [50, 55, 60])
weight['slim'] = fuzz.trimf(weight.universe, [55, 60, 65])
weight['optimal'] = fuzz.trimf(weight.universe, [60, 75, 90])
weight['fat'] = fuzz.trimf(weight.universe, [75, 90, 105])
weight['obesity'] = fuzz.trimf(weight.universe, [105, 150, 200])

# Water amount person should drink per day
water_need = ctrl.Consequent(np.arange(3, 10, 0.5), 'water_need')

# Automatically split to poor/average/good
water_need.automf(3) 

# set of rules
rule1 = ctrl.Rule(temperature['low'] | age['youth'] | weight['light'], water_need['poor'])
rule2 = ctrl.Rule(temperature['low'] | age['youth'] | weight['slim'], water_need['poor'])
rule3 = ctrl.Rule(temperature['low'] | age['youth'] | weight['optimal'], water_need['average'])
rule4 = ctrl.Rule(temperature['low'] | age['youth'] | weight['fat'], water_need['average'])
rule5 = ctrl.Rule(temperature['low'] | age['youth'] | weight['obesity'], water_need['good'])
rule6 = ctrl.Rule(temperature['low'] | age['adult'] | weight['light'], water_need['average'])
rule7 = ctrl.Rule(temperature['low'] | age['adult'] | weight['slim'], water_need['average'])
rule8 = ctrl.Rule(temperature['low'] | age['adult'] | weight['fat'], water_need['good'])
rule9 = ctrl.Rule(temperature['low'] | age['adult'] | weight['obesity'], water_need['good'])
rule10 = ctrl.Rule(temperature['low'] | age['old'], water_need['average'])
rule11 = ctrl.Rule(temperature['optimal'] | age['old'], water_need['good'])
rule12 = ctrl.Rule(temperature['optimal'] | age['youth'] | weight['light'], water_need['average'])
rule13 = ctrl.Rule(temperature['optimal'] | age['youth'] | weight['slim'], water_need['average'])
rule14 = ctrl.Rule(temperature['optimal'] | age['youth'] | weight['optimal'], water_need['average'])
rule15 = ctrl.Rule(temperature['optimal'] | age['youth'] | weight['fat'], water_need['good'])
rule16 = ctrl.Rule(temperature['optimal'] | age['youth'] | weight['obesity'], water_need['good'])
rule17 = ctrl.Rule(temperature['optimal'] | age['adult'] | weight['light'], water_need['average'])
rule18 = ctrl.Rule(temperature['optimal'] | age['adult'] | weight['slim'], water_need['average'])
rule19 = ctrl.Rule(temperature['optimal'] | age['adult'] | weight['optimal'], water_need['average'])
rule20 = ctrl.Rule(temperature['optimal'] | age['adult'] | weight['fat'], water_need['good'])
rule21 = ctrl.Rule(temperature['optimal'] | age['adult'] | weight['obesity'], water_need['good'])
rule22 = ctrl.Rule(temperature['high'], water_need['good'])

# rules
water_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22])

# simulation
person = ctrl.ControlSystemSimulation(water_ctrl)

# inputs

person.input['temperature'] = 15
person.input['weight'] = 89
person.input['age'] = 29

# do the magic
person.compute()

print("Quantity of water needed (liters): ", person.output['water_need'])
water_need.view(sim=person)

# display graph
plt.show()
