symptom_pred = [1,0,1,0,1,0,1,0,1,0]
        pred_nn = keras.models.load_model("symp_nn.h5")
        pred_rf = keras.models.load_model("symp_rf.h5")
        pred_1 = pred_nn.predict([symptoms_pred])
        pred_2 = pred_rf.predict([symptoms_pred])
        pred_3 = (pred_1 + pred_2) / 2
        print(pred_1,"********",pred_2,"********",pred_3,"********")
        