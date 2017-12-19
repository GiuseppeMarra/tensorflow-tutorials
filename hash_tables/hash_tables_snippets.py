import tensorflow as tf

sess = tf.InteractiveSession()
keys = tf.constant(["hello", "how", "are"], tf.string)
values = tf.constant([1., 2., 3.], tf.float32)
query = tf.constant(["how", "are"], tf.string)


#HashTable
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
out = table.lookup(query)
table.init.run()
print(out.eval())

#MutableHashTable
table = tf.contrib.lookup.MutableDenseHashTable(key_dtype = tf.string,
                                        value_dtype = tf.float32,
                                        default_value = -1,
                                        empty_key = tf.constant("pad"))


insert_op = table.insert(keys,values) #In the MutableHashTable the insert/update is an operation, since it can be run during learning
sess.run(insert_op)

ind = table.lookup(query)
ind = tf.cast(ind, tf.int64)

embeddings = tf.constant([[12.4, 17.2, 0.77], [1.11, 2.22, 3.77], [0.11, 4.39, 11.1], [90.11, 112.39, 1.1]])
out = tf.gather(params=embeddings, indices=ind)
print(sess.run(out))
