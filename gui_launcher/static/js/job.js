(function(){
  const el = document.getElementById('jobData');
  const jobId = el ? el.getAttribute('data-job-id') : null;
  if (!jobId) return;
  setTimeout(() => { try { window.location.href = `/job/${jobId}`; } catch (e) {} }, 700);
})();