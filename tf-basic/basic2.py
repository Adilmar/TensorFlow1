# Import tensorflow
import tensorflow as tf

# Inicializando as constantes
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiplicando 
result = tf.multiply(x1, x2)

# Inicializando seção
sess = tf.Session()

# Print do resultado
print(sess.run(result))

# Fechar a seção
sess.close()