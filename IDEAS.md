# Ideas


## Embedding

- Feature on nodes : not close to a wall / distance to wall
- skip

## Model

## Performance

- profiling : https://pytorch.org/docs/stable/bottleneck.html
=> "to" = loading is the bottleneck

## Levels

- very simple levels

## Debug

scatter_max in optimize => fixed

- FUCK GO BACK with an additional sink node

- ANNEALING epsilon


scatter_max verified => remove gradient tape ? ===> SEEMS TO CHANGE AND IMPROVE
issue : consider walls and self but their value are not updated ??!
===> remove wall from masks ?

issue : fixed epsilon ==> PERFORMANCES DETERIORATES DURING TRIANING !!!


issue : verify
# TODO Try to remove for loop
    for i in range(batch_size):
        state_action_values[i] = state_action_values_batch[state_batch.batch == i][
            action_batch[i]
        ]
==> seems OK


