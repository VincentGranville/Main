from sdv.demo import get_available_demos
from sdv.demo import load_tabular_demo
from sdv.tabular import CopulaGAN

demos = get_available_demos()
print(demos)  # show list of demo datasets

metadata, real_data = load_tabular_demo('student_placements_pii',metadata=True)
print("\nReal data:\n",real_data.head())
model = CopulaGAN()
model = CopulaGAN(primary_key='student_id',anonymize_fields={'address': 'address' })
model.fit(real_data)
synth_data1 = model.sample(200)
print("\nSynth. set 1:\n",synth_data1.head())

model.save('my_model.pkl')                # this shows how to save the model
loaded = CopulaGAN.load('my_model.pkl')   # load the model, and 
synth_data2 = loaded.sample(200)          # get new set of synth. data
print("\nSynth. set 2\n:",synth_data2.head())
