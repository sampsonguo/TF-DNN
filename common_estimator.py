#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import game_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--hash_bucket_size', default=100000, type=int,
                    help='hash bucket size for categorical column with hash bucket')
parser.add_argument('--hidden_units', default=[64, 16], type=list,
                    help='hidden units for dnn classifier')
parser.add_argument('--model_dir', default='./model', type=str,
                    help='model_dir for tensorflow')
parser.add_argument('--serving_model_dir', default='./serving_model', type=str,
                    help='model dir for tensorflow serving')
parser.add_argument('--classifier_mode', default='wide_and_deep', type=str,
                    help='dnn or wide or wide_and_deep')
parser.add_argument('--export_mode', default='raw', type=str,
                    help='raw or parsing')
parser.add_argument('--mode', default='train_and_eval', type=str,
                    help='train_and_eval or export_model')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = game_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    wide_feature_columns = []
    for key in train_x.keys():
        #my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        game_category_hashed_feature = tf.feature_column.categorical_column_with_hash_bucket(
            key = key,
            hash_bucket_size = args.hash_bucket_size,
            dtype = tf.int64)
        #game_indicator_column_feature = tf.feature_column.indicator_column(
        #    categorical_column = game_category_hashed_feature)
        game_embedding_feature= tf.feature_column.embedding_column(
            categorical_column = game_category_hashed_feature,
            dimension = 100,
            combiner = 'sqrtn',
            initializer = None,
            ckpt_to_load_from = None,
            tensor_name_in_ckpt = None,
            max_norm = None,
            trainable = True)
        my_feature_columns.append(game_embedding_feature)
        wide_feature_columns.append(game_category_hashed_feature)

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    if args.classifier_mode == "wide":
        classifier = tf.estimator.LinearClassifier(
            feature_columns = wide_feature_columns,
            model_dir = args.model_dir,
            n_classes = 2,
            weight_column = None,
            label_vocabulary = None,
            optimizer = 'Ftrl',
            config = None,
            partitioner = None,
            warm_start_from = None,
            loss_reduction = tf.losses.Reduction.SUM,
            sparse_combiner = 'sum')
        pass
    elif args.classifier_mode == "deep":
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=args.hidden_units,
            model_dir=args.model_dir,
            # The model must choose between 3 classes.
            n_classes=2)
    else:
        pass

    if args.mode == "train_and_eval":
        # Train the Model.
        classifier.train(
            input_fn=lambda:game_data.train_input_fn(train_x, train_y,
                                                     args.batch_size),
            steps=args.train_steps)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:game_data.eval_input_fn(test_x, test_y,
                                                    args.batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
    elif args.mode == "export_model":

        features_dict = {}
        # Export the model
        # Parsing Mode
        if args.export_mode == "parsing":
            serving_input_receiver = tf.estimator.export.build_parsing_serving_input_receiver_fn(
                feature_spec = None,
                default_batch_size = None)
        # Raw Mode
        else:
            features_dict["hash_imei"] = tf.placeholder(dtype = tf.int64, shape = (None, 100), name = 'hash_imei')
            features_dict["app_id"] = tf.placeholder(dtype = tf.int64, shape = (None, 1000000), name = 'app_id')
            features_dict["as_gs"] = tf.placeholder(dtype = tf.int64, shape = (None, 100), name = 'as_gs')
            serving_input_receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(
                features = features_dict,
                default_batch_size = None)

        classifier.export_savedmodel(
            export_dir_base = args.serving_model_dir,
            serving_input_receiver_fn = lambda: serving_input_receiver(),
            assets_extra = None,
            as_text = False,
            checkpoint_path = None)


    """
    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))
    """

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
