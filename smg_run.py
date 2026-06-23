import stormpy
import payntbind

smg_file = "models/smg/simple-game/sketch.templ"
prop = "<<circle>> R{\"rew\"}max=? [F \"goal\"]" # alternatively: <<0>> R{\"rew\"}max=? [F \"goal\"]

# Load the SMG model
program = stormpy.parse_prism_program(smg_file)
properties = stormpy.parse_properties_for_prism_program(prop, program, None)
model = stormpy.build_model(program, properties)

print(model)

result = payntbind.synthesis.model_check_smg(model, properties[0].raw_formula,
                                                        only_initial_states=False, set_produce_schedulers=True)

print(f"result at initial state: {result.at(0)}")