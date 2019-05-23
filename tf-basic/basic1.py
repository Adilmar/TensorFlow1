# Importando o tensorflow
import tensorflow as tf

# Inicializando as constantes 
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiplicando os Tensors
result = tf.multiply(x1, x2)

# Print the result
print(result)