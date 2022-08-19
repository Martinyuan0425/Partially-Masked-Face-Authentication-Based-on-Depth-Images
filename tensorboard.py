path = r"C:\Users\606\Desktop\facenet0523\facenet-tf2-main\logs\facenet_normal\train\events.out.tfevents.1653296154.DESKTOP-C05VQQL.5688.6862.v2"



from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
event_acc.Reload()

for scalar in event_acc.Tags()["scalars"]:
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar))
    print(scalar)
    print(vals)