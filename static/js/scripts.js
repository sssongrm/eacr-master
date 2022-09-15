(function ($) {
	"use strict";

	/*==========  Responsive Navigation  ==========*/
	$('.main-nav').children().clone().appendTo('.responsive-nav');
	$('.responsive-menu-open').on('click', function(event) {
		event.preventDefault();
		$('body').addClass('no-scroll');
		$('.responsive-menu').addClass('open');
		return false;
	});
	$('.responsive-menu-close').on('click', function(event) {
		event.preventDefault();
		$('body').removeClass('no-scroll');
		$('.responsive-menu').removeClass('open');
		return false;
	});
	$('.responsive-nav li').each(function(index) {
		if ($(this).find('ul').length) {
			var text = $(this).find('> a').text();
			var id = text.replace(/\s+/g, '-').toLowerCase();
			id = id.replace('&','');
			$(this).find('> a').attr('href', '#collapse-' + id);
			$(this).find('> a').attr('data-toggle', 'collapse');
			$(this).find('> a').append('<i class="fa fa-angle-down"></i>');
			$(this).find('> ul').attr('id', 'collapse-' + id);
			$(this).find('> ul').addClass('collapse');
		}
	});
	$('.responsive-nav a').on('click', function() {
		if ($(this).parent().hasClass('collapse-active')) {
			$(this).parent().removeClass('collapse-active');
		} else {
			$(this).parent().addClass('collapse-active');
		}
	});

	/*==========  Login / Signup  ==========*/
	$('.login-open').on('click', function(event) {
		event.preventDefault();
		$('.login-wrapper').addClass('open');
		$('.signup-wrapper').removeClass('open');
	});
	$(document).on('click', function(event) { 
		if (!$(event.target).closest('.login').length && !$(event.target).closest('.login-open').length) {
			$('.login-wrapper').removeClass('open');
		}
	});
	$('.signup-open').on('click', function(event) {
		event.preventDefault();
		$('.signup-wrapper').addClass('open');
		$('.login-wrapper').removeClass('open');
	});
	$(document).on('click', function(event) { 
		if (!$(event.target).closest('.signup').length && !$(event.target).closest('.signup-open').length) {
			$('.signup-wrapper').removeClass('open');
		}
	});

	/*==========  Accordion  ==========*/
	$('.panel-heading a').on('click', function() {
		if ($(this).parents('.panel-heading').hasClass('active')) {
			$('.panel-heading').removeClass('active');
		} else {
			$('.panel-heading').removeClass('active');
			$(this).parents('.panel-heading').addClass('active');
		}
	});

	/*==========  Highlights Slider  ==========*/
	$('.highlight-slider').owlCarousel({
		loop: true,
		nav: true,
		dots: false,
		navText: ['<i class="pe-7s-angle-left"></i>','<i class="pe-7s-angle-right"></i>'],
		items: 6,
		responsive: {
			0: {
				items: 2
			},
			480: {
				items: 6
			},
			769: {
				items: 6
			}
		}
	});

	/*==========  Directory Slider  ==========*/
	$('.directory-slider').owlCarousel({
		loop: false,
		nav: true,
		dots: false,
		navText: ['<i class="pe-7s-angle-left"></i>','<i class="pe-7s-angle-right"></i>'],
		items: 4,
		center: true,
		startPosition: 1,
		responsive: {
			0: {
				items: 1,
				startPosition: 0
			},
			480: {
				items: 2,
				startPosition: 0
			},
			769: {
				items: 4
			}
		}
	});

	/*==========  Directory Single Slider  ==========*/
	$('.directory-single-slider').owlCarousel({
		loop: true,
		nav: true,
		dots: false,
		navText: ['<i class="pe-7s-angle-left"></i>','<i class="pe-7s-angle-right"></i>'],
		items: 4,
		autoWidth: true,
		navContainer: '#customNav',
		responsive: {
			0: {
				items: 1,
				autoWidth: false,
				autoHeight: true
			},
			480: {
				items: 2
			},
			769: {
				items: 4
			}
		}
	});

	/*==========  Related products Slider  ==========*/

	$('.related-products-slider').owlCarousel({
		loop: false,
		nav: true,
		dots: false,
		navText: ['<i class="pe-7s-angle-left"></i>','<i class="pe-7s-angle-right"></i>'],
		items: 3,
		startPosition: 0,
		responsive: {
			0: {
				items: 1,
				startPosition: 0
			},
			480: {
				items: 2,
				startPosition: 0
			},
			769: {
				items: 3
			}
		}
	});

	/*==========  Blog Gallery  ==========*/
	$('.blog-gallery').owlCarousel({
		loop: true,
		nav: true,
		dots: false,
		navText: ['<i class="pe-7s-angle-left"></i>','<i class="pe-7s-angle-right"></i>'],
		items: 1
	});

	/* COUNTDOWN */
	$("#countdown").countdown({
		date: "1 Jan 2018 00:00:00", // Put your date here
		format: "on"
	});
	
	/*==========  Validate Email  ==========*/
	function validateEmail($validate_email) {
		var emailReg = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;
		if( !emailReg.test( $validate_email ) ) {
			return false;
		} else {
			return true;
		}
		return false;
	}
	
	/*==========  Contact Form  ==========*/
	$('#contact-form').on('submit', function() {
		$('#contact-form').find('.button').prop('disabled', true);
		if (validateEmail($('#contact-email').val()) && $('#contact-email').val().length !== 0 && $('#contact-name').val().length !== 0 && $('#contact-message').val().length !== 0) {
			var action = $(this).attr('action');
			$.ajax({
				type: "POST",
				url : action,
				data: {
					contact_name: $('#contact-name').val(),
					contact_email: $('#contact-email').val(),
					contact_subject: $('#contact-subject').val(),
					contact_message: $('#contact-message').val()
				},
				success: function() {
					$('#contact-form').find('.button').prop('disabled', false);
					swal({
						title: 'Success!',
						text: 'Thanks for contacting us!',
						type: 'success',
						html: true
					});
				},
				error: function() {
					$('#contact-form').find('.button').prop('disabled', false);
					swal({
						title: 'Error!',
						text: 'Sorry, an error occurred.',
						type: 'error',
						html: true
					});
				}
			});
		} else if (!validateEmail($('#contact-email').val()) && $('#contact-email').val().length !== 0 && $('#contact-name').val().length !== 0 && $('#contact-message').val().length !== 0) {
			$('#contact-form').find('.button').prop('disabled', false);
			swal({
				title: 'Oops!',
				text: 'Please enter a valid email.',
				html: true
			});
		} else {
			$('#contact-form').find('.button').prop('disabled', false);
			swal({
				title: 'Oops!',
				text: 'Please fill out all the fields.',
				html: true
			});
		}
		return false;
	});

	/*==========  Newsletter Form  ==========*/
	var $form = $('#mc-embedded-subscribe-form');
	$form.on('submit', function() {
		$form.find('.button').prop('disabled', true);
		if (validateEmail($('#mce-EMAIL').val()) && $('#mce-EMAIL').val().length !== 0) {
			$.ajax({
				type: $form.attr('method'),
				url: $form.attr('action'),
				data: $form.serialize(),
				cache: false,
				dataType: 'json',
				contentType: 'application/json; charset=utf-8',
				error: function(err) {
					$form.find('.button').prop('disabled', false);
					swal({
						title: 'Error!',
						text: err.msg,
						type: 'error',
						html: true
					});
				},
				success: function(data) {
					if (data.result !== 'success') {
						$form.find('.button').prop('disabled', false);
						swal({
							title: 'Wait!',
							text: data.msg,
							html: true
						});
					} else {
						$form.find('.button').prop('disabled', false);
						swal({
							title: 'Success!',
							text: data.msg,
							type: 'success',
							html: true
						});
					}
				}
			});
		} else {
			$form.find('.button').prop('disabled', false);
			swal({
				title: 'Error!',
				text: 'Please enter a valid email.',
				type: 'error',
				html: true
			});
		}
		return false;
	});

	$(document).on('click', function(event) { 
		if (!$(event.target).closest('.marker-wrapper').length) {
			$('.marker-wrapper').removeClass('open');
		}
	});

	/*==========  Video Popup  ==========*/

	$('.video-play').nivoLightbox({
		afterShowLightbox: function() {
			var src = $('.nivo-lightbox-content > iframe').attr('src');
			$('.nivo-lightbox-content > iframe').attr('src', src + '?autoplay=1');
			if ($(window).width() < 769) {
				var height = $('.nivo-lightbox-content iframe').height() / 2 + 28;
				$('.nivo-lightbox-close').css('margin-top', -height);
			}
		},
		beforeShowLightbox: function() {
			$('.video-play').find('i').removeClass('pe-7s-play');
			$('.video-play').find('i').addClass('pe-7s-pause');
			if ($('.bgAudio').length) {
				var player = $('.bgAudio');
				if (player[0].paused == false) {
					player.animate({volume: 0}, 1000, function() {
						player[0].pause();
					});
					$('.audioControl').addClass('pause');
				} else {
					player[0].play();
					player.animate({volume: 1}, 1000);
					$('.audioControl').removeClass('pause');
				}
			}
		},
		afterHideLightbox: function() {
			$('.video-play').find('i').removeClass('pe-7s-pause');
			$('.video-play').find('i').addClass('pe-7s-play');
			if ($('.bgAudio').length) {
				var player = $('.bgAudio');
				if (player[0].paused == false) {
					player.animate({volume: 0}, 1000, function() {
						player[0].pause();
					});
					$('.audioControl').addClass('pause');
				} else {
					player[0].play();
					player.animate({volume: 1}, 1000);
					$('.audioControl').removeClass('pause');
				}
			}
		}
	});

	/*==========  Load more button  ==========*/
	$(".blog-grid .row").hide();
		$(".blog-grid .row").slice(0, 3).show();
		$("#blog-load-more").on('click', function (e) {
		e.preventDefault();
		$(".blog-grid .row:hidden").slice(0, 1).slideDown();
		if ($(".blog-grid .row:hidden").length == 0) {
			$("#blog-load-more").fadeOut('slow');
		}
	});
	$(".products .col-sm-6").hide();
		$(".products .col-sm-6").slice(0, 8).show();
		$("#products-load-more").on('click', function (e) {
		e.preventDefault();
		$(".products .col-sm-6:hidden").slice(0, 2).slideDown();
		if ($(".products .col-sm-6:hidden").length == 0) {
			$("#products-load-more").fadeOut('slow');
		}
	});
	
	$('#price-slider').noUiSlider({
		connect: true,
		behaviour: 'tap',
		margin: 100,
		start: [39, 350],
		step: 1,
		range: {
			'min': 0,
			'max': 500
		}
	});
	$('#price-slider').Link('lower').to($('#price-min'), null, wNumb({
		decimals: 2
	}));
	$('#price-slider').Link('upper').to($('#price-max'), null, wNumb({
		decimals: 2
	}));

})(jQuery);