from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from dp_few_shot_learning_system_example_level_single import MAMLFewShotClassifierExampleSingle
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset


# Combines the arguments, model, data and experiment builders to run an experiment

args, device = get_args()
maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader
dataset_name = args.dataset_name
# args.seed = 2

if 'imagenet' in dataset_name:
    model = MAMLFewShotClassifierExampleSingle(args=args, device=device, im_shape=(2, args.image_channels,
                                               args.image_height, args.image_width))
else:
    model = MAMLFewShotClassifierExampleSingle(args=args, device=device,
                                               im_shape=(2, args.image_channels,
                                                         args.image_height,
                                                         args.image_width))

maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device, example_level=True)
# maml_system.random_initialize_test()
maml_system.run_rdp_compute_example_level(eps_target=2.5, dataset=dataset_name)
maml_system.run_experiment()
maml_system.run_experiment_test_only()



