import loss
import tensorflow as tf
import unittest

class TestLoss(unittest.TestCase):
  """Class used to test the loss function defined in loss.py"""

  def test_correct(self):
    y_actual = tf.convert_to_tensor([[1,0],[0,1], [1,1]])
    y_predicted = tf.convert_to_tensor([[1.0,0],[0,1], [1,1]])

    loss_out = loss.custom_crossentropy(y_actual, y_predicted)
    loss_out = tf.round((10**3)*loss_out)/10**3
    self.assertEqual(loss_out, tf.zeros(()))

  def test_incorrect(self):
    y_actual = tf.convert_to_tensor([[1,0]])
    y_predicted = tf.convert_to_tensor([[0.5,0]])

    loss_out = loss.custom_crossentropy(y_actual, y_predicted)
    loss_out = tf.round((10**3)*loss_out)/10**3
    comparison = tf.convert_to_tensor(-1*tf.math.log(0.5))
    comparison = tf.round((10**3)*comparison)/comparison
    self.assertEqual(loss_out, comparison)

  def test_alpha(self):
    y_actual = tf.convert_to_tensor([[1,0]])
    y_predicted = tf.convert_to_tensor([[0.5,0]])

    loss_out = loss.custom_crossentropy(y_actual, y_predicted, alpha=0.7)
    loss_out = tf.round((10**3)*loss_out)/10**3
    comparison = tf.convert_to_tensor(-1*2*0.7*tf.math.log(0.5))
    comparison = tf.round((10**3)*comparison)/comparison
    self.assertEqual(loss_out, comparison)

  def test_epsilon(self):
    y_actual = tf.convert_to_tensor([[1,0],[0,1], [1,1]])
    y_predicted = tf.convert_to_tensor([[0.0,1],[1,0], [0,0]])

    loss_out = loss.custom_crossentropy(y_actual, y_predicted, epsilon=1)
    self.assertEqual(loss_out, tf.zeros(()))



if __name__ == "__main__":
  unittest.main()