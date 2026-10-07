/* D2S Bharat - core statistics in SAS (for SAS Viya for Learners / SAS Studio).
   Reproduces the main RQ2/RQ3 analyses and key RQ1 preparation steps.
   STATUS: written to mirror the Python analysis; not yet executed in SAS - run in VFL and
   compare with the report tables before quoting.
   Upload the four SAS files to your VFL home folder first and set &path. */

%let path=/home/&sysuserid/hackathon;

/* ---------- JDS: junior skills -> salary hike (RQ2) ---------- */
proc import datafile="&path/JDS Skill Traits.xlsx" out=jds dbms=xlsx replace;
  getnames=yes;
run;
data jds;
  set jds;
  rename 'maths-stats_skills'n = maths_stats_skills;
run;

proc means data=jds n mean std min median max maxdec=2;
  class salary_hike_high_or_low;
  var big_data_skills maths_stats_skills coding_skills ai_and_ml_skills
      dashboard_and_storytelling_skills;
run;

/* Mann-Whitney (Wilcoxon rank-sum) per skill */
proc npar1way data=jds wilcoxon;
  class salary_hike_high_or_low;
  var big_data_skills maths_stats_skills coding_skills ai_and_ml_skills
      dashboard_and_storytelling_skills;
run;

/* Standardise, then logistic regression with odds ratios per +1 SD and ROC */
proc stdize data=jds out=jds_z method=std;
  var big_data_skills maths_stats_skills coding_skills ai_and_ml_skills
      dashboard_and_storytelling_skills;
run;
proc logistic data=jds_z plots(only)=roc;
  model salary_hike_high_or_low(event='1') = big_data_skills maths_stats_skills coding_skills
        ai_and_ml_skills dashboard_and_storytelling_skills / clodds=wald;
run;

/* Depth-3 decision tree */
proc hpsplit data=jds maxdepth=3 minleafsize=10 seed=42;
  class salary_hike_high_or_low;
  model salary_hike_high_or_low = big_data_skills maths_stats_skills coding_skills
        ai_and_ml_skills dashboard_and_storytelling_skills;
  partition fraction(validate=0.3 seed=42);
run;

/* ---------- SDS: senior traits -> success (RQ3) ---------- */
proc import datafile="&path/SDS Personality Traits.xlsx" out=sds dbms=xlsx replace;
  getnames=yes;
run;
data sds;
  set sds;
  rename ' extraversion'n = extraversion
         'success_ classification_ high_low'n = success_high_low;
run;
proc npar1way data=sds wilcoxon;
  class success_high_low;
  var neuroticism extraversion openness_to_experience agreeableness conscientiousness;
run;
proc stdize data=sds out=sds_z method=std;
  var neuroticism extraversion openness_to_experience agreeableness conscientiousness;
run;
proc logistic data=sds_z plots(only)=roc;
  model success_high_low(event='1') = neuroticism extraversion openness_to_experience
        agreeableness conscientiousness / clodds=wald;
run;
proc hpsplit data=sds maxdepth=3 minleafsize=10 seed=42;
  class success_high_low;
  model success_high_low = neuroticism extraversion openness_to_experience agreeableness
        conscientiousness;
run;

/* ---------- Analytics Jobs: de-duplication and derived variables (RQ1 prep) ---------- */
proc import datafile="&path/Analytics Jobs.csv" out=aj dbms=csv replace;
  getnames=yes; guessingrows=max;
run;
/* exact duplicates ignoring the row id */
proc sort data=aj(drop=s_no) out=aj_dedup nodupkey dupout=aj_dups;
  by _all_;
run;
data aj_clean;
  set aj_dedup;
  exp_min = input(scan(experience, 1, '-'), best.);
  exp_max = input(scan(scan(experience, 2, '-'), 1, ' '), best.);
  sal_min = input(scan(salary, 1, 't'), best.);
  sal_max = input(scan(salary, 2, 'o'), best.);
  length band $8;
  band = salary;
  band_rank = whichc(salary, '0to3', '3to6', '6to10', '10to15', '15to25', '25to50');
  job_type_clean = lowcase(strip(job_type));
  if job_type_clean = 'analytic' then job_type_clean = 'analytics';
  has_sql = (index(lowcase(key_skills), 'sql') > 0);
  has_sas = (prxmatch('/\bsas\b/i', key_skills) > 0);
run;
proc freq data=aj_clean;
  tables band_rank job_type_clean / missing;
run;
/* ordinal (cumulative logit) model of salary band on experience + example skill flags */
proc logistic data=aj_clean;
  model band_rank = exp_min has_sql has_sas / link=clogit clodds=wald;
run;
