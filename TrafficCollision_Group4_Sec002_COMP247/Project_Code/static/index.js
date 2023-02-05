let model = {};
const COLUMN_COUNT = 4;
const model_names = ['Support Vector Classifier', 'RandomForest', 'Neural Network'];
const model_ids = ['svm-model', 'rf-model', 'nn-model'];
let model_name = '';
let columns = [];
let values = [];
async function loadData() {
   // const res = await fetch('/scores/RandomForest');
   // model = await res.json();
   // columns = model['Columns'];
   // values = model['Values'];
   //selectModel(0);
   // selectModel(1);
    selectModel(2);
    //selectModel(3);
    //selectModel(4);

    
}

function createForm() {
    var html = "";
    var num = 0;
    
    while (1) {
        html += `<div class="row">`;
        console.log(columns.length);
        for (var i = 0; i < COLUMN_COUNT; i++) {
            if (num >= columns.length)
                break;

            let c = columns[num];
            html += `<div class="col">
                        <label for="${c}">${c}:</label><br>
                        <input type="text"  id="${c}" name="${c}" value="${values[num]}" />
                    </div>`
            num++;
            
        }
        html += `</div>`;
        if (num >= columns.length)
            break;
    }

    document.getElementById("form").innerHTML = html;
}

async function selectModel(num) {
    console.log(model);
    const name = model_names[num];
    model_name = name;
   console.log(name);
   let str=null;

   switch(name){
    
    case "Support Vector Classifier":
         str='/scores/SVC';
         break;
    case "RandomForest":
        str='/scores/RandomForest';
        break;
    case "Neural Network":
        str='/scores/NeuralNet';
        break;
   }
   
   model=await call(str);
   //console.log(model);
    columns = model['Columns'];
    values = model['Values'];
    console.log(columns);
   // console.log(values);
   createForm();
   
   
   $('#graph-title').text("Displaying metrics for " + name);
    $('#form-title').text(name);

    // set active
    for (var i = 0; i < 3; i++) {
        if (i == num)
            $('#' + model_ids[i]).addClass("active");
        else
            $('#' + model_ids[i]).removeClass("active");
    }

    const data = model;
    console.log(data);
    if (!data)
        return;
    
    

    const accuracy = parseFloat(data["accuracy"]) * 100;
    const precision = parseFloat(data["precision"]) * 100;
    const recall = parseFloat(data["recall"]) * 100;
    const f1 = parseFloat(data["f1"]) * 100;

    //console.log(f1);


    $('#accuracy').asPieProgress('go', accuracy);
    $('#precision').asPieProgress('go', precision);
    $('#recall').asPieProgress('go', recall);
    $('#f1_score').asPieProgress('go', f1);

    

}

async function call(str)
{
    
    const res1=await fetch(str)
    model = await res1.json();
    console.log(str);
    return model;
}

function selectTab(num) {
    if (num == 1) {
        $('#prediction').css('color', '#1171d7');
        $('#prediction').css('border-bottom', '2px solid #1171d7');

        $('#scores').css('color', '#aeafb0');
        $('#scores').css('border-bottom', 'none');
        $('#graph-title').hide();
        $('.graph').hide();
        $('.form').show();
    }
    else {
        $('#scores').css('color', '#1171d7');
        $('#scores').css('border-bottom', '2px solid #1171d7');

        $('#prediction').css('color', '#aeafb0');
        $('#prediction').css('border-bottom', 'none');

        $('#graph-title').show();
        $('.graph').show();
        $('.form').hide();
    }
}

async function predict() {
    var data = {};
    //var values1 = [-8855709.28228975,5429786.06691169,2012,18,"Major Arterial","Etobicoke York",43.769145,-79.55219,"Intersection","At Intersection","No Control"
//,"Clear","Dark, artificial","Dry", "Rear End","Passenger","5 to 9","Minimal","East","Other","Going Ahead","Driving Properly","Normal","Yes","Yes", "D31",21];
    
  var values=[];
 for (var i = 0; i < columns.length; i++) {
        c = columns[i];
         var v = document.getElementById(c).value;
        values.push(v);
     }
    data['values'] = values;
    data['model'] = model_name;
    data['columns'] = columns;

    console.log("Values in predict:"+values);
    console.log(data);

    switch(model_name){
        
        case "Support Vector Classifier":
             str='/predict/SVC';
             break;
        case "RandomForest":
            str='/predict/RandomForest';
            break;
        case "Neural Network":
            str='/predict/NeuralNet';
            break;
       }

    //const json=await call2(str);
    console.log(str);
    const res = await fetch(str, {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });

    console.log("printing request"+res);
    const json = await res.json();
    console.log(json);
    //return json;
    document.getElementById("recovered").innerHTML = `Predicted Outptut: ${json.prediction}`;
}




jQuery(function ($) {
    $('.pie_progress').asPieProgress({
        namespace: 'pie_progress'
    });

    //console.log("hi");
    // get model data
    loadData();
    
    selectTab(1);
});