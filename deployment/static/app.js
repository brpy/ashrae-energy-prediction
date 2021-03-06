const app = new Vue({
  el:'#app',
  delimiters: ['[[', ']]'],
  data:{
    errors:[],
    site:null,
    building:null,
    meter:0,
    timestamp: "2017/09/24 21:00:00",
    air_temperature: null,
    cloud_coverage: null,
    dew_temperature: null,
    precip_depth_1_hr: null,
    sea_level_pressure: null,
    wind_direction: null,
    wind_speed: null,
    sites : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    building_range :[[0,104],[105,155],[156,290],[291,564],[565,655],[656,744],[745,788],[789,803],[804,873],[874,997],[998,1027],[1028,1032],[1033,1068],[1069,1222],[1223,1324],[1325,1448]]

  },
  methods:{
    checkForm:function(e) {
      if(this.meter!==null && this.building!==null && this.timestamp!==null && !isNaN(Number(this.air_temperature)) && !isNaN(Number(this.cloud_coverage)) && !isNaN(Number(this.dew_temperature)) && !isNaN(Number(this.precip_depth_1_hr)) && !isNaN(Number(this.sea_level_pressure)) && !isNaN(Number(this.wind_direction)) && !isNaN(Number(this.wind_speed)) && this.timestamp.split(/\/| |:/).length==6) return true;
      
      this.errors = [];
      if(this.meter==null) this.errors.push("Meter required.");
      if(this.building==null) this.errors.push("Building Id required.");
      if(this.timestamp==null) this.errors.push("Timestamp required.");
      if(this.timestamp.split(/\/| |:/).length!==6) this.errors.push("Timestamp not in format yyyy/mm/dd HH:mm:ss");
      if(isNaN(Number(this.air_temperature))) this.errors.push(`"${this.air_temperature}" is not a valid input for air_temperature. Required float`);
      if(isNaN(Number(this.cloud_coverage))) this.errors.push(`"${this.cloud_coverage}" is not a valid input for cloud_coverage. Required float`);
      if(isNaN(Number(this.dew_temperature))) this.errors.push(`"${this.dew_temperature}" is not a valid input for dew_temperature. Required float`);
      if(isNaN(Number(this.precip_depth_1_hr))) this.errors.push(`"${this.precip_depth_1_hr}" is not a valid input for precip_depth_1_hr. Required float`);
      if(isNaN(Number(this.sea_level_pressure))) this.errors.push(`"${this.sea_level_pressure}" is not a valid input for sea_level_pressure. Required float`);
      if(isNaN(Number(this.wind_direction))) this.errors.push(`"${this.wind_direction}" is not a valid input for wind_direction. Required float`);
      if(isNaN(Number(this.wind_speed))) this.errors.push(`"${this.wind_speed}" is not a valid input for wind_speed. Required float`);

      e.preventDefault();
    },
    buildingLimit:function(){
        if(!this.site) return null;
        start = this.building_range[this.site][0];
        end = this.building_range[this.site][1];
        arr = [];
        for (i=start;i<=end;i++){
            arr.push(i)
        } 
        return arr;
    }
    
  }
})

